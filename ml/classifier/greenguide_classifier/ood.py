"""open-set OOD reject — 클래스 prototype(임베딩 중심) 구축·보정·거리 계산.

추론 시: 입력 임베딩이 모든 클래스 prototype 에서 멀면(cosine 거리 > τ), softmax 가
확신해도 '기타/분류 불가'로 reject. 마우스 등 학습 범위 밖(OOD)을 차단.
softmax 는 "11개 중 최선"만 보지만, 임베딩 거리는 "학습된 무엇과도 닮았는가"를 본다.

prototype 은 모델 버전마다 임베딩 공간이 바뀌므로 export 직후 재계산해야 한다.
서빙(`embedding` ONNX 출력)과 동일 전처리/공간을 쓰도록 ONNX 로 계산.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from greenguide_common import imaging, settings
from greenguide_common.logging import fail_open, get_logger
from PIL import Image

from greenguide_classifier import config
from greenguide_classifier.dataset import load_manifest
from greenguide_classifier.infer import load_session
from greenguide_classifier.split import load_splits, subset_items

log = get_logger(__name__)

ONNX_PATH: Path = config.MODELS_DIR / "cnn" / "classifier.onnx"
PROTO_PATH: Path = config.MODELS_DIR / "cnn" / "prototypes.npz"

_MEAN = imaging.MEAN_ARRAY
_STD = imaging.STD_ARRAY
_SAMPLE_PER_CLASS = 100
_DEFAULT_PERCENTILE = 97.5   # τ = in-distribution 거리의 이 분위수


def _abs(source_path: str) -> str:
    return os.path.join(config.PREPROCESSOR_ROOT, source_path)


def _prep(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def _session() -> ort.InferenceSession:
    return load_session(ONNX_PATH)


def _embed(paths: list[str], sess: ort.InferenceSession, batch: int = 64) -> np.ndarray:
    """이미지 경로 → L2 정규화된 임베딩 (N, 512). 디코드 실패분은 건너뜀."""
    out: list[np.ndarray] = []
    buf: list[np.ndarray] = []

    def flush() -> None:
        if not buf:
            return
        x = np.stack(buf).astype(np.float32)
        emb = sess.run(["embedding"], {"image": x})[0]
        out.append(emb)
        buf.clear()

    for p in paths:
        try:
            buf.append(_prep(p))
        except Exception:  # noqa: BLE001 — fail-open: 손상 이미지 건너뜀
            continue
        if len(buf) >= batch:
            flush()
    flush()
    if not out:
        return np.zeros((0, 512), dtype=np.float32)
    e = np.concatenate(out, axis=0)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def build_prototypes(sample_per_class: int = _SAMPLE_PER_CLASS) -> dict[str, np.ndarray]:
    """train 데이터에서 클래스별 임베딩 평균(L2 정규화) → prototypes.npz 저장."""
    items = load_manifest()
    try:
        train = subset_items(items, load_splits()["train"])
    except Exception:  # noqa: BLE001 — fail-open: splits 없으면 전체 manifest 사용
        train = items
    by_label: dict[str, list[str]] = {}
    for it in train:
        by_label.setdefault(it["label"], []).append(_abs(it["source_path"]))

    sess = _session()
    rng = np.random.default_rng(config.SPLIT_SEED)
    labels = sorted(by_label)
    centroids = []
    for lab in labels:
        paths = by_label[lab]
        idx = rng.permutation(len(paths))[:sample_per_class]
        emb = _embed([paths[i] for i in idx], sess)
        if len(emb) == 0:
            centroids.append(np.zeros(512, dtype=np.float32))
            continue
        c = emb.mean(axis=0)
        centroids.append(c / (np.linalg.norm(c) + 1e-8))
        log.info(f"prototype[{lab}] ← {len(emb)} samples")
    arr = np.stack(centroids).astype(np.float32)
    PROTO_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PROTO_PATH, labels=np.array(labels), centroids=arr)
    log.info(f"prototypes 저장 → {PROTO_PATH} ({len(labels)} 클래스)")
    return {"labels": labels, "centroids": arr}


def load_prototypes() -> tuple[list[str], np.ndarray]:
    data = np.load(PROTO_PATH, allow_pickle=True)
    return list(data["labels"]), data["centroids"]


def nearest_distance(emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """L2 정규화된 emb (N,512) → 가장 가까운 prototype 까지 cosine 거리 (N,)."""
    sims = emb @ centroids.T          # (N, K) cosine 유사도
    return 1.0 - sims.max(axis=1)     # 거리 = 1 - 최대유사도


def calibrate(percentile: float = _DEFAULT_PERCENTILE) -> dict:
    """val 의 nearest-distance 분포 → τ 추천 + (가능하면) 실사용 OOD 비교."""
    labels, centroids = load_prototypes()
    items = load_manifest()
    val = subset_items(items, load_splits()["val"])
    sess = _session()
    emb = _embed([_abs(it["source_path"]) for it in val], sess)
    d = nearest_distance(emb, centroids)

    pcts = {p: float(np.percentile(d, p)) for p in (50, 90, 95, 97.5, 99)}
    tau = float(np.percentile(d, percentile))
    print("[ood] in-distribution(val) nearest-distance 분위수:")
    for p, v in pcts.items():
        print(f"   p{p}: {v:.4f}")
    print(f"[ood] 추천 τ (p{percentile}) = {tau:.4f}")
    print(f"   → 정상 {percentile:.0f}% 통과, 그보다 먼 입력은 OOD reject")

    # 실사용 OOD 후보(user_uploads etc 피드백) 와 비교 — 네트워크 가능 시
    with fail_open(log, "실사용 OOD 비교"):
        import io

        import requests

        from etc_queue import _supabase  # 재사용
        rows = (_supabase().table(settings.SUPABASE_TABLE_USER_UPLOADS)
                .select("image_url,feedback_label")
                .eq("feedback_label", "etc").limit(20).execute()).data or []
        ood_emb = []
        for r in rows:
            try:
                im = Image.open(io.BytesIO(requests.get(r["image_url"], timeout=15).content))
                im = im.convert("RGB").resize((224, 224), Image.BILINEAR)
                a = (np.asarray(im, np.float32) / 255.0 - _MEAN) / _STD
                x = np.ascontiguousarray(a.transpose(2, 0, 1))[None]
                e = sess.run(["embedding"], {"image": x})[0]
                ood_emb.append(e[0] / (np.linalg.norm(e[0]) + 1e-8))
            except Exception:  # noqa: BLE001 — fail-open: 다운로드/디코딩 실패 건너뜀
                continue
        if ood_emb:
            od = nearest_distance(np.stack(ood_emb), centroids)
            caught = int((od > tau).sum())
            print(f"[ood] 실사용 etc 피드백 {len(od)}장 거리: "
                  f"min {od.min():.3f} / 중앙 {np.median(od):.3f} / max {od.max():.3f}")
            print(f"   τ={tau:.3f} 로 {caught}/{len(od)} 가 OOD reject 됨")

    return {"tau": tau, "percentiles": pcts}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("build", "all"):
        build_prototypes()
    if cmd in ("calibrate", "all"):
        calibrate()
