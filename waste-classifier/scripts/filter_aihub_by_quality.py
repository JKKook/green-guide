"""AI Hub crop 품질 점수화 — u2netp 으로 단일 객체·중앙 dominance 측정.

배경(2026-05-30 분석): AI Hub 의 paper 30장 표본 중 약 60% 가 라벨 객체가 주된
객체가 아닌 facility scene crop(검은 바닥 + 잡종 폐기물). 모델이 학습 노이즈로
이를 흡수해 "어두운 잡배경 = paper" 같은 spurious feature 를 만들 수 있음.

이 스크립트는 1차 게이트: u2netp 의 가장 salient 한 객체가 (a) 충분히 크고
(b) 중앙에 가깝다면 "single-subject 구도" 로 통과. 라벨 일치성은 검증하지 못함
(가장 salient 한 게 라벨된 클래스 객체인지는 모름) — 그건 모델 agreement 로 별도.

기본은 dry-run: 통계 + 경계 후보 spot 만 출력. --apply 시 통과한 파일만 남기는
keep 디렉토리에 심볼릭 링크. 원본은 절대 안 건드림.

사용:
    .venv/bin/python scripts/filter_aihub_by_quality.py --class paper
    .venv/bin/python scripts/filter_aihub_by_quality.py --class paper \\
        --min-area 0.18 --max-offset 0.25 --apply
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"
U2NETP_PATH = PROJECT_ROOT.parent / "waste-api" / "models" / "u2netp.onnx"
CLASSIFIER_PATH = PROJECT_ROOT / "outputs" / "models" / "cnn" / "classifier.onnx"

# u2netp 입출력 사양
_SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)
_MASK_THRESHOLD = 0.30   # saliency 0~1 에서 객체로 간주할 하한

# 분류기 입력 (ImageNet 정규화, 224)
_CLF_SIZE = 224
_CLF_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_CLF_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_u2netp() -> ort.InferenceSession:
    if not U2NETP_PATH.exists():
        sys.exit(f"u2netp 없음: {U2NETP_PATH}")
    return ort.InferenceSession(str(U2NETP_PATH), providers=["CPUExecutionProvider"])


# ─── CLIP zero-shot (외부 심판자) ───────────────────────────────
# 우리 모델이 AI Hub 시각 노이즈를 paper feature 로 흡수해버려 self-filter 가
# closed loop 가 되는 문제를 외부 시각 지식(CLIP) 으로 깸.
_CLIP_PROMPTS = {
    "cardboard":   "a photo of a corrugated cardboard box",
    "clothes":     "a photo of clothes or fabric items like shirts",
    "electronics": "a photo of electronic devices or appliances",
    "etc":         "a photo of an unidentifiable miscellaneous object",
    "food_waste":  "a photo of food waste or leftover food scraps",
    "glass":       "a photo of glass bottles or glass jars",
    "metal":       "a photo of metal cans or aluminum waste",
    "non_object":  "a photo of an empty surface, hand, or background only",
    "paper":       "a photo of paper, newspaper, or paper documents",
    "plastic":     "a photo of plastic bottles or plastic containers",
    "styrofoam":   "a photo of white styrofoam or polystyrene packaging",
    "trash":       "a photo of mixed garbage or general non-recyclable trash",
    "vinyl":       "a photo of a plastic bag or thin plastic film",
}


def load_clip():
    """CLIP 로드 + 모든 클래스 텍스트 임베딩을 미리 계산해 반환."""
    import torch  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
    name = "openai/clip-vit-base-patch32"
    print(f"  [clip] loading {name}...")
    proc = CLIPProcessor.from_pretrained(name)
    mdl = CLIPModel.from_pretrained(name).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl = mdl.to(device)
    labels = sorted(_CLIP_PROMPTS.keys())
    prompts = [_CLIP_PROMPTS[c] for c in labels]
    with torch.no_grad():
        ti = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        txt_emb = mdl.get_text_features(**ti)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return mdl, proc, txt_emb, labels, device


def clip_probs(mdl, proc, txt_emb, device, img_path: Path) -> np.ndarray | None:
    import torch  # noqa: PLC0415
    try:
        img = Image.open(img_path).convert("RGB")
        ii = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            ie = mdl.get_image_features(**ii)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            # CLIP 표준 스케일 (100*) 후 softmax
            logits = (100.0 * ie @ txt_emb.T)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()
        return probs
    except Exception:  # noqa: BLE001
        return None


def load_classifier() -> tuple[ort.InferenceSession, list[str]]:
    if not CLASSIFIER_PATH.exists():
        sys.exit(f"classifier 없음: {CLASSIFIER_PATH}")
    sess = ort.InferenceSession(str(CLASSIFIER_PATH), providers=["CPUExecutionProvider"])
    # config 에서 라벨 순서 가져옴 (manifest 기반 — 학습과 동기)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src import config  # noqa: PLC0415
    config.refresh_classes_from_manifest()
    return sess, list(config.CLASS_LABELS)


def _preprocess_clf(img: Image.Image) -> np.ndarray:
    im = img.convert("RGB").resize((_CLF_SIZE, _CLF_SIZE), Image.BILINEAR)
    arr = (np.array(im).astype(np.float32) / 255.0 - _CLF_MEAN) / _CLF_STD
    chw = arr.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return chw


def model_probs(sess: ort.InferenceSession, img_path: Path) -> np.ndarray | None:
    """이미지 → softmax 확률 벡터 (NUM_CLASSES,) 또는 실패 시 None."""
    try:
        img = Image.open(img_path)
        inp = _preprocess_clf(img)
    except Exception:  # noqa: BLE001
        return None
    name = sess.get_inputs()[0].name
    logits = sess.run(None, {name: inp})[0][0]  # (NUM_CLASSES,)
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _preprocess(img: Image.Image) -> np.ndarray:
    im = img.convert("RGB").resize((_SIZE, _SIZE), Image.LANCZOS)
    arr = np.array(im).astype(np.float64)
    mx = arr.max()
    if mx > 0:
        arr = arr / mx
    arr = (arr - _MEAN) / _STD
    chw = arr.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return chw


def saliency_metrics(session: ort.InferenceSession, img_path: Path) -> dict:
    """단일 이미지 → {area, cx, cy, offset, blob_count}.

    - area: saliency 가 임계값 넘는 픽셀의 비율 (0~1)
    - cx, cy: 객체 무게중심 (0~1 정규화)
    - offset: 중심과 (0.5, 0.5) 사이 정규화 거리 (0=완벽 중앙, ~0.7=대각 코너)
    - blob_count: 마스크의 연결 성분 수 (1=단일, 2+=다객체/잡종)
    """
    try:
        img = Image.open(img_path)
        inp = _preprocess(img)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    name = session.get_inputs()[0].name
    out = session.run(None, {name: inp})[0][0, 0]   # (320, 320)
    mi, ma = float(out.min()), float(out.max())
    if ma - mi > 1e-8:
        out = (out - mi) / (ma - mi)
    else:
        return {"area": 0.0, "cx": 0.5, "cy": 0.5, "offset": 0.0, "blob_count": 0}

    mask = out >= _MASK_THRESHOLD
    area = float(mask.mean())
    if area < 1e-4:
        return {"area": 0.0, "cx": 0.5, "cy": 0.5, "offset": 0.0, "blob_count": 0}

    ys, xs = np.where(mask)
    cy, cx = float(ys.mean()) / _SIZE, float(xs.mean()) / _SIZE
    offset = float(np.hypot(cx - 0.5, cy - 0.5))

    # 연결 성분 수 — 단순 BFS 로 카운트 (성능 위해 16x16 다운샘플)
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((16, 16), Image.NEAREST)
    sm = (np.array(small) > 0).astype(np.uint8)
    blob_count = _count_blobs(sm)

    return {
        "area": round(area, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "offset": round(offset, 4),
        "blob_count": blob_count,
    }


def _count_blobs(grid: np.ndarray) -> int:
    """4-연결 BFS 로 1 의 연결 성분 수 카운트."""
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i, j] and not seen[i, j]:
                count += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= h or x < 0 or x >= w:
                        continue
                    if seen[y, x] or not grid[y, x]:
                        continue
                    seen[y, x] = True
                    stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Hub crop 품질 점수화 (u2netp)")
    ap.add_argument("--class", dest="cls", required=True, help="클래스 slug (예: paper)")
    ap.add_argument("--min-area", type=float, default=0.18,
                    help="객체 면적 비율 하한 (기본 0.18 = 프레임의 18%)")
    ap.add_argument("--max-offset", type=float, default=0.25,
                    help="객체 중심 ↔ 프레임 중심 거리 상한 (0~0.71, 기본 0.25)")
    ap.add_argument("--max-blobs", type=int, default=2,
                    help="허용 연결 성분 수 (기본 2 — 단일 + 약간의 잡티 OK)")
    ap.add_argument("--model-check", action="store_true",
                    help="활성 분류기로 agreement 점수까지 계산 (manifest 에 기록)")
    ap.add_argument("--min-label-prob", type=float, default=0.0,
                    help="모델이 라벨 클래스에 부여한 확률의 하한 (model-check 시)")
    ap.add_argument("--max-other-confident", type=float, default=1.01,
                    help="다른 클래스가 이 확률 이상 + 라벨보다 높으면 'mislabel' 로 거절")
    ap.add_argument("--clip-check", action="store_true",
                    help="CLIP zero-shot 으로 외부 라벨 일치성 점수 측정 (manifest 기록)")
    ap.add_argument("--min-clip-label-prob", type=float, default=0.0,
                    help="CLIP 가 라벨 클래스에 부여한 확률의 하한 (clip-check 시)")
    ap.add_argument("--manifest",
                    default=str(PROJECT_ROOT / "diagnostics" / "filter_manifest.csv"),
                    help="결과 manifest CSV 경로")
    ap.add_argument("--apply", action="store_true",
                    help="설정 시 통과 파일을 keep_aihub_{class}/ 에 심볼릭 링크")
    ap.add_argument("--limit", type=int, default=0,
                    help="N>0 이면 처음 N장만 (테스트용)")
    args = ap.parse_args()

    cls_dir = RAW_DIR / args.cls
    if not cls_dir.exists():
        sys.exit(f"클래스 디렉토리 없음: {cls_dir}")

    files = sorted(p for p in cls_dir.iterdir()
                   if p.name.startswith("aihub_") and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"aihub_* 파일 없음: {cls_dir}")

    print(f"[filter] {args.cls}: AI Hub 파일 {len(files)}장")
    print(f"  기준: area ≥ {args.min_area}, offset ≤ {args.max_offset}, blobs ≤ {args.max_blobs}")
    print(f"  manifest: {args.manifest}")

    sess = load_u2netp()
    clf_sess: ort.InferenceSession | None = None
    clf_labels: list[str] = []
    label_index = -1
    if args.model_check:
        clf_sess, clf_labels = load_classifier()
        if args.cls not in clf_labels:
            sys.exit(f"클래스 '{args.cls}' 가 모델 라벨에 없음: {clf_labels}")
        label_index = clf_labels.index(args.cls)
        print(f"  model-check ON: paper index={label_index}, "
              f"min_label_prob={args.min_label_prob}, "
              f"max_other_confident={args.max_other_confident}")

    clip_state = None
    clip_label_index = -1
    if args.clip_check:
        mdl, proc, txt_emb, clip_labels, clip_dev = load_clip()
        if args.cls not in clip_labels:
            sys.exit(f"클래스 '{args.cls}' 가 CLIP 라벨에 없음: {clip_labels}")
        clip_label_index = clip_labels.index(args.cls)
        clip_state = (mdl, proc, txt_emb, clip_labels, clip_dev)
        print(f"  clip-check ON: paper index={clip_label_index}, "
              f"device={clip_dev}, min_clip_label_prob={args.min_clip_label_prob}")

    rows = []
    pass_count = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        m = saliency_metrics(sess, f)
        if "error" in m:
            rows.append({"file": f.name, "decision": "error",
                         **{k: "" for k in ("area", "cx", "cy", "offset", "blob_count")},
                         "model_top1": "", "model_top1_prob": "", "model_label_prob": "",
                         "clip_top1": "", "clip_top1_prob": "", "clip_label_prob": "",
                         "reason": m["error"]})
            continue

        # u2netp 게이트
        reasons = []
        if m["area"] < args.min_area:
            reasons.append(f"area<{args.min_area}")
        if m["offset"] > args.max_offset:
            reasons.append(f"offset>{args.max_offset}")
        if m["blob_count"] == 0:
            reasons.append("no_object")
        elif m["blob_count"] > args.max_blobs:
            reasons.append(f"blobs>{args.max_blobs}")
        u2_ok = not reasons

        # 모델 agreement (옵션)
        model_top1 = ""
        model_top1_prob = ""
        model_label_prob = ""
        if clf_sess is not None:
            probs = model_probs(clf_sess, f)
            if probs is None:
                reasons.append("model_error")
            else:
                top_idx = int(probs.argmax())
                model_top1 = clf_labels[top_idx]
                model_top1_prob = round(float(probs[top_idx]), 4)
                model_label_prob = round(float(probs[label_index]), 4)
                # 거절 규칙 1: 라벨 클래스의 확률이 너무 낮음
                if probs[label_index] < args.min_label_prob:
                    reasons.append(f"label_prob<{args.min_label_prob}")
                # 거절 규칙 2: 다른 클래스가 강하게 확신 + 라벨보다 높음
                other_max = float(probs[np.arange(len(probs)) != label_index].max())
                if other_max >= args.max_other_confident and other_max > probs[label_index]:
                    reasons.append(f"other_confident>={args.max_other_confident}")

        # CLIP zero-shot agreement (옵션)
        clip_top1 = ""
        clip_top1_prob = ""
        clip_label_prob = ""
        if clip_state is not None:
            mdl, proc, txt_emb, clip_labels, clip_dev = clip_state
            cprobs = clip_probs(mdl, proc, txt_emb, clip_dev, f)
            if cprobs is None:
                reasons.append("clip_error")
            else:
                top_idx = int(cprobs.argmax())
                clip_top1 = clip_labels[top_idx]
                clip_top1_prob = round(float(cprobs[top_idx]), 4)
                clip_label_prob = round(float(cprobs[clip_label_index]), 4)
                if cprobs[clip_label_index] < args.min_clip_label_prob:
                    reasons.append(f"clip_label_prob<{args.min_clip_label_prob}")

        ok = u2_ok and (clf_sess is None or all(
            not r.startswith(("label_prob", "other_confident", "model_error")) for r in reasons
        )) and (clip_state is None or all(
            not r.startswith(("clip_label_prob", "clip_error")) for r in reasons
        ))
        rows.append({"file": f.name, **m,
                     "model_top1": model_top1,
                     "model_top1_prob": model_top1_prob,
                     "model_label_prob": model_label_prob,
                     "clip_top1": clip_top1,
                     "clip_top1_prob": clip_top1_prob,
                     "clip_label_prob": clip_label_prob,
                     "decision": "pass" if ok else "fail",
                     "reason": "" if ok else ",".join(reasons)})
        if ok:
            pass_count += 1

        if i % 200 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(files)}] pass={pass_count} "
                  f"({100*pass_count/i:.1f}%) elapsed={elapsed:.0f}s")

    # manifest 저장
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "file", "decision", "area", "cx", "cy", "offset", "blob_count",
            "model_top1", "model_top1_prob", "model_label_prob",
            "clip_top1", "clip_top1_prob", "clip_label_prob", "reason",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ manifest 저장: {out}")

    # 통계 요약
    valid = [r for r in rows if "error" not in (r.get("reason") or "")
             and r["decision"] in ("pass", "fail")]
    if valid:
        areas = [r["area"] for r in valid if isinstance(r["area"], (int, float))]
        offsets = [r["offset"] for r in valid if isinstance(r["offset"], (int, float))]
        blobs = [r["blob_count"] for r in valid if isinstance(r["blob_count"], int)]
        print("\n[통계 — area]")
        print(f"  min={min(areas):.3f} p25={np.percentile(areas, 25):.3f} "
              f"med={np.median(areas):.3f} p75={np.percentile(areas, 75):.3f} max={max(areas):.3f}")
        print("[통계 — offset]")
        print(f"  min={min(offsets):.3f} p25={np.percentile(offsets, 25):.3f} "
              f"med={np.median(offsets):.3f} p75={np.percentile(offsets, 75):.3f} max={max(offsets):.3f}")
        print("[통계 — blob_count]")
        for b in sorted(set(blobs)):
            print(f"  {b} blobs: {blobs.count(b)}장 ({100*blobs.count(b)/len(blobs):.1f}%)")
        # 거절 사유
        fail_reasons: dict[str, int] = {}
        for r in valid:
            if r["decision"] == "fail":
                for x in r["reason"].split(","):
                    fail_reasons[x] = fail_reasons.get(x, 0) + 1
        print("\n[거절 사유 (다중 카운트 가능)]")
        for k, v in sorted(fail_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}장")
    print(f"\n결과: {pass_count}/{len(files)} pass ({100*pass_count/len(files):.1f}%)")

    if args.apply:
        keep_dir = RAW_DIR.parent / f"keep_aihub_{args.cls}"
        keep_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r.get("decision") == "pass":
                src = cls_dir / r["file"]
                dst = keep_dir / r["file"]
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
        print(f"\n✓ {pass_count}장 심볼릭 링크 → {keep_dir}")
    else:
        print("\n(dry-run) --apply 없이 실행 — manifest 만 저장됨")
    return 0


if __name__ == "__main__":
    sys.exit(main())
