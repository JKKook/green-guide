#!/usr/bin/env python3
"""계층 모델용 임베딩 프로토타입 + OOD 임계 보정.

src/ood.py 의 원칙(softmax 는 '최선'만, 임베딩 거리는 '닮았는가'를 봄)을
계층 ONNX(embedding 512d 출력)에 적용:

1. train 의 fine-감독 아이템에서 클래스당 최대 CAP 장 샘플 → embedding 추출
2. 클래스별 L2-정규화 평균 = prototype
3. val 아이템의 최근접 prototype cosine distance 분포에서 97.5퍼센타일 = τ
4. outputs/models/cnn_hier/ood.npz (+ taxonomy.json 에 τ 기록)

실행: .venv/bin/python scripts/build_hier_prototypes.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.hier_dataset import (  # noqa: E402
    HierImageDataset,
    build_hier_items,
    load_or_build_hier_splits,
)

MODELS_DIR = config.MODELS_DIR / "cnn_hier"
ONNX_PATH = MODELS_DIR / "classifier.onnx"
OOD_PATH = MODELS_DIR / "ood.npz"
SIDECAR = MODELS_DIR / "taxonomy.json"

PER_CLASS_CAP = 400        # prototype 표본 상한 (충분 + 빠름)
VAL_SAMPLE_CAP = 4000      # τ 보정용 val 표본
TAU_PERCENTILE = 97.5      # in-distribution 97.5% 를 통과시키는 임계
SEED = 42


def _extract_embeddings(sess: ort.InferenceSession, items: list[dict]) -> np.ndarray:
    loader = DataLoader(
        HierImageDataset(items, augment=False),
        batch_size=config.CNN_BATCH_SIZE, shuffle=False,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )
    embs = []
    for x, _, _ in loader:
        (emb,) = sess.run(["embedding"], {"image": x.numpy()})
        embs.append(emb)
    out = np.concatenate(embs, axis=0)
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


def main() -> None:
    rng = np.random.default_rng(SEED)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    train_items = [items[i] for i in splits["train"] if items[i]["sup_kind"] == "fine"]
    val_items = [items[i] for i in splits["val"] if items[i]["sup_kind"] == "fine"]

    # 클래스별 샘플링
    by_class: dict[str, list[dict]] = defaultdict(list)
    for it in train_items:
        by_class[it["sup_slug"]].append(it)
    proto_items: list[dict] = []
    for slug, pool in by_class.items():
        idx = rng.permutation(len(pool))[:PER_CLASS_CAP]
        proto_items.extend(pool[i] for i in idx)
    print(f"prototype 표본: {len(proto_items):,} ({len(by_class)} classes)")

    embs = _extract_embeddings(sess, proto_items)
    protos: dict[str, np.ndarray] = {}
    for slug in by_class:
        mask = np.array([it["sup_slug"] == slug for it in proto_items])
        m = embs[mask].mean(axis=0)
        protos[slug] = m / np.clip(np.linalg.norm(m), 1e-9, None)

    # τ 보정: val in-distribution 의 최근접 거리 분포
    val_idx = rng.permutation(len(val_items))[:VAL_SAMPLE_CAP]
    val_sample = [val_items[i] for i in val_idx]
    val_embs = _extract_embeddings(sess, val_sample)
    P = np.stack([protos[s] for s in sorted(protos)])       # (C, 512)
    sims = val_embs @ P.T                                    # cosine sim
    min_dist = 1.0 - sims.max(axis=1)
    tau = float(np.percentile(min_dist, TAU_PERCENTILE))
    print(f"val 최근접 거리: median={np.median(min_dist):.4f}, "
          f"p97.5={tau:.4f} → τ={tau:.4f}")

    slugs_sorted = sorted(protos)
    np.savez_compressed(
        OOD_PATH,
        prototypes=np.stack([protos[s] for s in slugs_sorted]),
        slugs=np.array(slugs_sorted),
        tau=np.array([tau]),
    )
    print(f"→ {OOD_PATH}")

    # 사이드카에 τ 기록 (서빙이 함께 로드)
    sc = json.loads(SIDECAR.read_text(encoding="utf-8"))
    sc["ood"] = {"tau": tau, "percentile": TAU_PERCENTILE, "file": "ood.npz"}
    SIDECAR.write_text(json.dumps(sc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {SIDECAR} (ood.tau 갱신)")


if __name__ == "__main__":
    main()
