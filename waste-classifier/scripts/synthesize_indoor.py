"""실내 배경 합성 엔진 — Kaggle 객체 + 실내 배경 → 사용자 분포 모사 학습 데이터.

설계: DATA_AUGMENTATION_DESIGN.md §6, §11

흐름:
1. Kaggle 객체 풀에서 무작위 객체 선택 (라벨 정합성 95%+)
2. u2netp 으로 객체 alpha mask 추출
3. 실내 배경 풀에서 무작위 배경 선택
4. 합성: 광원 매칭 → 그림자 → 객체 placement → albumentations 후처리
5. JPEG q=90 저장 + 메타데이터 jsonl 기록

사용:
    .venv/bin/python scripts/synthesize_indoor.py \\
        --our-class etc --n 100 [--bg-pool data/raw/_aux/backgrounds] \\
        [--seed 42] [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"
SYNTH_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "synthetic_indoor"
AUX_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "_aux"
U2NETP_PATH = PROJECT_ROOT.parent / "waste-api" / "models" / "u2netp.onnx"
SYNTHESIS_VERSION = "v1"

CANVAS_SIZE = 320   # 합성 캔버스 (학습은 224 로 resize)
JPEG_QUALITY = 90

# u2netp 정규화
_U2NET_SIZE = 320
_U2NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
_U2NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# 합성 후처리 (폰 카메라 도메인 randomization)
SYNTHESIS_AUGMENT = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.10, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.RandomGamma(gamma_limit=(85, 115), p=0.3),
    A.ImageCompression(quality_lower=75, quality_upper=95, p=0.5),
    A.MotionBlur(blur_limit=5, p=0.15),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
    A.ISONoise(intensity=(0.05, 0.20), p=0.2),
])


def load_u2netp() -> ort.InferenceSession:
    if not U2NETP_PATH.exists():
        sys.exit(f"u2netp 없음: {U2NETP_PATH}")
    return ort.InferenceSession(str(U2NETP_PATH), providers=["CPUExecutionProvider"])


def extract_alpha(sess: ort.InferenceSession, img: Image.Image) -> np.ndarray:
    """u2netp 으로 객체 saliency alpha 추출 (320x320, 0~1 float32)."""
    im = img.convert("RGB").resize((_U2NET_SIZE, _U2NET_SIZE), Image.LANCZOS)
    arr = np.array(im).astype(np.float64)
    mx = arr.max()
    if mx > 0:
        arr = arr / mx
    arr = (arr - _U2NET_MEAN) / _U2NET_STD
    chw = arr.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: chw})[0][0, 0]
    mi, ma = float(out.min()), float(out.max())
    if ma - mi > 1e-8:
        out = (out - mi) / (ma - mi)
    return out.astype(np.float32)


def find_object_pool(our_class: str) -> list[Path]:
    """객체 풀 — Kaggle/user 우선, electronics 같이 Kaggle 없는 클래스는 AI Hub fallback.

    품질은 u2netp alpha + quality_check() 로 사후 필터링.
    합성 자체가 noise 있는 객체는 거절하므로, 풀이 다양한 게 더 좋음.
    """
    cls_dir = RAW_DIR / our_class
    if not cls_dir.exists():
        return []
    kaggle_user, aihub_only = [], []
    for p in cls_dir.iterdir():
        if not p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            continue
        n = p.name
        # synth_, taco_ 등 합성·외부 출처 자체 사용 X
        if n.startswith(("synth_", "taco_")):
            continue
        # Kaggle 또는 user 출처가 우선
        if n.startswith(("kg2_", "user_")) or \
           (n.startswith(our_class) and n[len(our_class):].rstrip(".jpg").rstrip(".jpeg").isdigit()):
            kaggle_user.append(p)
        elif n.startswith("aihub_"):
            aihub_only.append(p)
    if kaggle_user:
        return sorted(kaggle_user)
    # Kaggle/user 없으면 AI Hub fallback (electronics 같은 경우)
    return sorted(aihub_only)


def find_backgrounds(bg_pool_dir: Path) -> list[Path]:
    if not bg_pool_dir.exists():
        return []
    return sorted(p for p in bg_pool_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def compose(
    obj_img: Image.Image, alpha: np.ndarray,
    bg_img: Image.Image, rng: np.random.Generator,
) -> Image.Image | None:
    """객체 + 배경 합성. 실패 시 None."""
    # 1. 배경 리사이즈
    bg = np.asarray(bg_img.convert("RGB").resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS),
                    dtype=np.float32)

    # 2. 객체 — alpha 가 큰 영역만 crop
    obj_arr = np.asarray(obj_img.convert("RGB").resize((CANVAS_SIZE, CANVAS_SIZE), Image.BILINEAR),
                         dtype=np.float32)
    mask = alpha > 0.4
    ys, xs = np.where(mask)
    if len(ys) < 200:
        return None  # 객체 너무 작음
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    obj_crop = obj_arr[y0:y1, x0:x1]
    alpha_crop = alpha[y0:y1, x0:x1]

    # 3. 회전 (±15°)
    rotation = float(rng.uniform(-15, 15))
    if abs(rotation) > 0.5:
        h, w = obj_crop.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
        obj_crop = cv2.warpAffine(obj_crop, M, (w, h), borderValue=0)
        alpha_crop = cv2.warpAffine(alpha_crop, M, (w, h), borderValue=0)

    # 4. 스케일 (객체 = 캔버스의 40~70%)
    scale_frac = float(rng.uniform(0.40, 0.70))
    target = int(CANVAS_SIZE * scale_frac)
    obj_resized = cv2.resize(obj_crop, (target, target), interpolation=cv2.INTER_AREA)
    alpha_resized = cv2.resize(alpha_crop, (target, target), interpolation=cv2.INTER_LINEAR)
    alpha_resized = np.clip(alpha_resized, 0, 1)

    # 5. 광원 매칭 (객체 톤을 배경 톤으로 약하게 끌어옴)
    bg_mean = bg.mean(axis=(0, 1))
    a3 = alpha_resized[..., None]
    obj_mean = (obj_resized * a3).sum(axis=(0, 1)) / (a3.sum() + 1e-6)
    tint = (bg_mean - obj_mean) * 0.15
    obj_tinted = np.clip(obj_resized + tint, 0, 255)

    # 6. 객체 placement (중앙 편향)
    cx = float(rng.uniform(0.20, 0.45))
    cy = float(rng.uniform(0.20, 0.45))
    px = int(cx * (CANVAS_SIZE - target))
    py = int(cy * (CANVAS_SIZE - target))

    # 7. 그림자 — gaussian-blurred alpha 를 배경에 부분 darkening
    shadow = cv2.GaussianBlur(alpha_resized.astype(np.float32), (15, 15), sigmaX=8) * 0.35
    shadow_offset = (5, 5)  # 우하단으로
    sy = py + shadow_offset[1]
    sx = px + shadow_offset[0]
    se_y = min(sy + target, CANVAS_SIZE)
    se_x = min(sx + target, CANVAS_SIZE)
    sh_h = se_y - sy
    sh_w = se_x - sx

    result = bg.copy()
    if sh_h > 0 and sh_w > 0:
        result[sy:se_y, sx:se_x] *= (1 - shadow[:sh_h, :sh_w, None])

    # 8. 객체 alpha blend
    pe_y = min(py + target, CANVAS_SIZE)
    pe_x = min(px + target, CANVAS_SIZE)
    oh = pe_y - py
    ow = pe_x - px
    if oh > 0 and ow > 0:
        obj_region = obj_tinted[:oh, :ow]
        a_region = alpha_resized[:oh, :ow, None]
        result[py:pe_y, px:pe_x] = obj_region * a_region + result[py:pe_y, px:pe_x] * (1 - a_region)

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 9. albumentations 후처리
    augmented = SYNTHESIS_AUGMENT(image=result)["image"]

    return Image.fromarray(augmented)


def quality_check(synthesized: Image.Image, alpha: np.ndarray) -> bool:
    """합성 결과 품질 게이트. 통과해야 학습용으로 채택."""
    arr = np.array(synthesized)
    # 너무 어둡거나 단조롭지 않은지
    mean_v = arr.mean()
    std_v = arr.std()
    if mean_v < 20 or mean_v > 240:
        return False  # 너무 어둡거나 밝음
    if std_v < 15:
        return False  # 단조로움
    # 객체가 실제로 합성됐는지 (alpha 면적 충분)
    if alpha.mean() < 0.05:
        return False  # 객체 알파가 너무 작음
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="실내 합성 엔진")
    ap.add_argument("--our-class", required=True, help="합성할 클래스 (예: etc)")
    ap.add_argument("--n", type=int, default=100, help="합성 장수 목표")
    ap.add_argument("--bg-pool", default=str(AUX_DIR / "backgrounds"),
                    help="실내 배경 풀 디렉토리")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="저장 안 하고 시험만")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + hash(args.our_class) % 100000)

    # 객체 풀
    obj_pool = find_object_pool(args.our_class)
    if not obj_pool:
        sys.exit(f"Kaggle 객체 없음: {args.our_class}")
    print(f"[synth] {args.our_class} — Kaggle 객체 {len(obj_pool)}장")

    # 배경 풀
    bg_pool = find_backgrounds(Path(args.bg_pool))
    if not bg_pool:
        sys.exit(f"배경 풀 비어 있음: {args.bg_pool}\n"
                 f"배경 풀 구축 스크립트 먼저 실행하세요")
    print(f"[synth] 배경 풀 {len(bg_pool)}장")

    # u2netp
    sess = load_u2netp()

    # 출력 디렉토리
    out_dir = SYNTH_DIR / args.our_class
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = SYNTH_DIR / "_manifest.jsonl"

    # 기존 합성 카운트 (이어쓰기 지원)
    existing = sorted(out_dir.glob(f"synth_{args.our_class}_*.jpg")) if out_dir.exists() else []
    start_idx = len(existing)
    print(f"  기존 {start_idx}장, 목표 {args.n}장 추가")

    saved = 0
    attempts = 0
    rejected = 0
    t0 = time.time()
    manifest_entries = []

    while saved < args.n and attempts < args.n * 5:  # 최대 5배 시도
        attempts += 1
        obj_path = obj_pool[rng.integers(0, len(obj_pool))]
        bg_path = bg_pool[rng.integers(0, len(bg_pool))]

        try:
            obj_img = Image.open(obj_path)
            bg_img = Image.open(bg_path)
        except Exception:  # noqa: BLE001
            continue

        alpha = extract_alpha(sess, obj_img)
        if alpha.mean() < 0.03:
            rejected += 1
            continue

        result = compose(obj_img, alpha, bg_img, rng)
        if result is None or not quality_check(result, alpha):
            rejected += 1
            continue

        seq = start_idx + saved
        out_name = f"synth_{args.our_class}_{seq:06d}.jpg"
        if not args.dry_run:
            result.save(out_dir / out_name, "JPEG", quality=JPEG_QUALITY)
            manifest_entries.append({
                "synth_id": out_name.replace(".jpg", ""),
                "class": args.our_class,
                "object_source": obj_path.name,
                "background_source": bg_path.name,
                "synthesis_version": SYNTHESIS_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
        saved += 1

        if saved % 20 == 0 or saved == args.n:
            elapsed = time.time() - t0
            print(f"  [{saved}/{args.n}] attempts={attempts} rejected={rejected} "
                  f"elapsed={elapsed:.0f}s ({saved/elapsed:.1f} img/s)")

    if not args.dry_run and manifest_entries:
        with manifest_path.open("a") as fh:
            for e in manifest_entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"\n✓ {saved}장 합성 → {out_dir}")
        print(f"  manifest 추가: {manifest_path}")
    else:
        print(f"\n(dry-run) 합성만 진행, 저장 안 함")

    print(f"\n총 시도: {attempts}, 채택: {saved}, 거절: {rejected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
