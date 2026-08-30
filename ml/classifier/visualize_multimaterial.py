"""PoC — CAM 기반 약지도 다중재질 분할.

아이디어: 모델의 per-class CAM (num_classes, 7, 7) 에서, 각 공간 셀마다
클래스 간 softmax → argmax 하면 "이 셀은 어떤 재질" 지도가 나온다.
u2netp 객체 mask 로 배경 셀은 제외. 인접 같은 재질을 묶어 영역별 라벨.

검증 목적: 한 물체 안에서 재질이 다른 영역(예: 페트병 몸통 vs 라벨)이
실제로 갈리는지 시각적으로 확인. 갈리면 다중재질 기능으로 발전 가능.

사용:
    .venv/bin/python visualize_multimaterial.py --image <path>
    .venv/bin/python visualize_multimaterial.py --label plastic --n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms
from waste_common import imaging, settings
from waste_common.logging import get_logger

from src import config
from src.model import CamWasteClassifierCNN, WasteClassifierCNN

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "cnn" / "best.pt"
U2NETP_PATH = settings.API_ROOT / "models" / "u2netp.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multimaterial"

_NORM = transforms.Normalize(list(imaging.IMAGENET_MEAN), list(imaging.IMAGENET_STD))
_PRE = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    _NORM,
])

# u2netp
_U2_MEAN = imaging.IMAGENET_MEAN
_U2_STD = imaging.IMAGENET_STD


def _load_cnn(device):
    model = WasteClassifierCNN(pretrained=False)
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    sd = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    cam_model = CamWasteClassifierCNN(model).to(device).eval()
    return cam_model


def _object_mask(u2_session, pil: Image.Image, grid: int) -> np.ndarray:
    """u2netp → grid×grid 객체 점유 비율 (0~1)."""
    if u2_session is None:
        return np.ones((grid, grid), dtype=np.float32)
    im = pil.convert("RGB").resize((320, 320), Image.LANCZOS)
    ary = np.array(im).astype(np.float64)
    mx = ary.max()
    if mx > 0:
        ary = ary / mx
    tmp = np.zeros((320, 320, 3))
    for c in range(3):
        tmp[:, :, c] = (ary[:, :, c] - _U2_MEAN[c]) / _U2_STD[c]
    inp = tmp.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    out = u2_session.run(None, {u2_session.get_inputs()[0].name: inp})[0][0, 0]
    mi, ma = out.min(), out.max()
    out = (out - mi) / (ma - mi + 1e-8)
    # grid 로 down-pool (평균)
    m = Image.fromarray((out * 255).astype(np.uint8)).resize((grid, grid), Image.BILINEAR)
    return np.array(m).astype(np.float32) / 255.0


def _softmax_axis0(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=0, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=0, keepdims=True)


def analyze(image_path: Path, cam_model, u2_session, out_path: Path,
            mask_thresh: float = 0.35, conf_thresh: float = 0.35):
    device = next(cam_model.parameters()).device
    pil = Image.open(image_path).convert("RGB")
    x = _PRE(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, cam = cam_model(x)  # cam: (1, C, h, w)
    cam = cam[0].cpu().numpy()      # (C, h, w)
    C, gh, gw = cam.shape
    probs_global = torch.softmax(logits[0], 0).cpu().numpy()
    global_idx = int(probs_global.argmax())

    # 셀별 클래스 분포 (softmax across classes)
    cell_probs = _softmax_axis0(cam)         # (C, h, w)
    cell_class = cell_probs.argmax(axis=0)   # (h, w)
    cell_conf = cell_probs.max(axis=0)       # (h, w)

    # 객체 mask
    omask = _object_mask(u2_session, pil, gh)  # (gh, gw)

    # 유효 셀: 객체 영역 + confidence 충분
    valid = (omask >= mask_thresh) & (cell_conf >= conf_thresh)

    # 영역별 클래스 집계
    from collections import Counter
    region_counts = Counter()
    for r in range(gh):
        for c in range(gw):
            if valid[r, c]:
                region_counts[int(cell_class[r, c])] += 1

    # 시각화 — 원본 | 셀별 재질 오버레이
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(pil)
    axes[0].set_title(f"input: {image_path.name}\nglobal: {config.CLASS_LABELS[global_idx]} "
                      f"({probs_global[global_idx]*100:.0f}%)", fontsize=10)
    axes[0].axis("off")

    # 오버레이: 셀별 클래스 색 + 라벨
    axes[1].imshow(pil)
    ih, iw = np.array(pil).shape[:2]
    ch, cw = ih / gh, iw / gw
    cmap = plt.get_cmap("tab10")
    for r in range(gh):
        for c in range(gw):
            if not valid[r, c]:
                continue
            cls = int(cell_class[r, c])
            color = cmap(cls % 10)
            rect = plt.Rectangle((c*cw, r*ch), cw, ch, facecolor=color,
                                 alpha=0.45, edgecolor="white", linewidth=0.5)
            axes[1].add_patch(rect)
    # 영역별 라벨 텍스트 (상위 클래스들)
    summary = " · ".join(
        f"{config.CLASS_LABELS[cls]}={n}" for cls, n in region_counts.most_common()
    )
    n_classes_found = len(region_counts)
    axes[1].set_title(f"CAM-argmax 영역 ({gh}x{gw}, {n_classes_found}개 재질)\n{summary}",
                      fontsize=9)
    axes[1].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return n_classes_found, region_counts


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--label", type=str)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device: {device}, classes: {list(config.CLASS_LABELS)}")
    if not CKPT_PATH.exists():
        sys.exit(f"체크포인트 없음: {CKPT_PATH}")
    cam_model = _load_cnn(device)
    u2 = ort.InferenceSession(str(U2NETP_PATH), providers=["CPUExecutionProvider"]) \
        if U2NETP_PATH.exists() else None
    log.info(f"u2netp: {'OK' if u2 else '없음 (객체 mask 생략)'}")

    if args.image:
        images = [args.image]
    else:
        raw = (settings.PREPROCESSOR_ROOT / "data" / "raw"
               / "garbage-classification" / args.label)
        if not raw.exists():
            sys.exit(f"라벨 폴더 없음: {raw}")
        imgs = sorted(p for p in raw.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        step = max(1, len(imgs) // args.n)
        images = imgs[::step][:args.n]

    for img in images:
        out = args.out / f"mm_{img.stem}.png"
        n, counts = analyze(img, cam_model, u2, out)
        print(f"  {img.name}: {n}개 재질 영역 → {out.relative_to(PROJECT_ROOT)}")
    print(f"\n✓ {args.out.relative_to(PROJECT_ROOT)}/ 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
