#!/usr/bin/env python3
"""Phase 3 — 다중객체 장면 합성 → 서빙 분포 정렬 학습 크롭 생성.

문제: 서빙은 u2netp bbox 크롭을 분류하는데, 혼재 장면의 크롭엔 이웃 물건
파편·배경이 섞임. 학습 데이터(단일 객체 타이트 크롭)에는 그런 샘플이 없어
분포 불일치 → 실사용 오답.

방법:
1. staging 크롭에서 클래스별 객체 풀 → u2netp 알파 컷아웃 (사전 계산)
2. 실내 배경 위에 2~3개 객체 합성 (그림자 + 위치/스케일 랜덤)
   - 60%: 혼동쌍 그룹 내 조합 (carton↔유리색상↔paper_cup 하드네거티브)
   - 40%: 전 클래스 랜덤 조합
3. 각 객체의 bbox+25% 패딩으로 캔버스를 크롭 (이웃 파편 포함!) →
   해당 객체의 라벨로 fine-staging/<staging_dir>/synmo_*.jpg 저장

주의: synmo_* 는 hier_dataset 이 frozen test 에서 제외 (train/val 전용).
실행: .venv/bin/python scripts/synthesize_multiobject.py [--per-class 800]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from _base import make_parser
from greenguide_common import imaging, settings
from PIL import Image, ImageFilter

from greenguide_classifier.infer import load_session

STAGING_CROPS = settings.WASTE_ROOT / "ml" / "data" / "raw" / "aihub_71385" / "crops"
BACKGROUNDS = (settings.PREPROCESSOR_ROOT / "data" / "raw"
               / "_aux" / "backgrounds")
FINE_STAGING = (settings.PREPROCESSOR_ROOT / "data" / "raw"
                / "fine-staging")
U2NETP = settings.API_ROOT / "models" / "u2netp.onnx"

CANVAS = (640, 480)
OBJ_SCALE = (0.30, 0.52)      # 캔버스 짧은변 대비 객체 크기
CROP_PAD = 0.25               # 학습 크롭 패딩 (이웃 파편 포함 목적)
CUTOUT_POOL_PER_CLASS = 250   # 클래스당 사전 컷아웃 수
STORE_MAX = 256

# 장면 구성 클래스 (staging dir 이름) — fine-staging 디렉터리와 일치해야 함
CLASSES = [
    "paper_pack", "paper_cup", "glass_clear", "glass_brown", "glass_green",
    "glass_deposit", "battery", "pet", "styrofoam_white", "light_bulb",
]
# 혼동쌍 하드네거티브 그룹 (fine_confusion_report 근거)
HARD_GROUPS = [
    ["paper_pack", "paper_cup", "glass_clear", "glass_brown"],   # carton 역혼동
    ["glass_clear", "glass_green", "glass_deposit", "pet"],      # 유리 가족 + PET
]

_U2_MEAN = imaging.IMAGENET_MEAN
_U2_STD = imaging.IMAGENET_STD


class Cutter:
    def __init__(self) -> None:
        self.sess = load_session(U2NETP)
        self.inp = self.sess.get_inputs()[0].name

    def cutout(self, img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
        """RGB → (RGBA 컷아웃, 알파 bbox). 실패 시 None."""
        im = img.convert("RGB").resize((320, 320), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float64)
        mx = a.max()
        if mx > 0:
            a = a / mx
        for c in range(3):
            a[:, :, c] = (a[:, :, c] - _U2_MEAN[c]) / _U2_STD[c]
        x = a.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0, 0]
        mi, ma = float(out.min()), float(out.max())
        if ma - mi < 1e-8:
            return None
        mask = (out - mi) / (ma - mi)
        if (mask > 0.4).mean() < 0.10:
            return None
        alpha320 = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
        alpha = alpha320.resize(img.size, Image.LANCZOS)
        rgba = img.convert("RGBA")
        rgba.putalpha(alpha)
        m = np.array(alpha) > 100
        ys, xs = np.where(m)
        if len(xs) < 100:
            return None
        return rgba, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def build_pool(cutter: Cutter, rng: random.Random) -> dict[str, list]:
    pool: dict[str, list] = {}
    for cls in CLASSES:
        d = STAGING_CROPS / cls
        files: list[Path] = []
        for cond in ("clean",):
            cd = d / cond
            if cd.exists():
                files.extend(cd.glob("*.jpg"))
        rng.shuffle(files)
        cuts = []
        for f in files[: CUTOUT_POOL_PER_CLASS * 2]:
            if len(cuts) >= CUTOUT_POOL_PER_CLASS:
                break
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            c = cutter.cutout(img)
            if c is not None:
                cuts.append(c)
        pool[cls] = cuts
        print(f"  {cls:16} 컷아웃 {len(cuts)}")
    return pool


def paste_object(canvas: Image.Image, cut: tuple, rng: random.Random,
                 occupied: list) -> tuple[int, int, int, int] | None:
    rgba, (bx0, by0, bx1, by1) = cut
    obj = rgba.crop((bx0, by0, bx1, by1))
    short = min(CANVAS)
    target = rng.uniform(*OBJ_SCALE) * short
    scale = target / max(obj.size)
    nw, nh = max(24, int(obj.width * scale)), max(24, int(obj.height * scale))
    obj = obj.resize((nw, nh), Image.LANCZOS)

    for _ in range(12):  # 겹침 최소화 배치 시도
        x = rng.randint(0, CANVAS[0] - nw)
        y = rng.randint(0, CANVAS[1] - nh)
        box = (x, y, x + nw, y + nh)
        overlap = any(
            not (box[2] < o[0] + 20 or o[2] < box[0] + 20
                 or box[3] < o[1] + 20 or o[3] < box[1] + 20)
            for o in occupied
        )
        if not overlap:
            break
    else:
        return None

    # 그림자 (알파 블러 → 오프셋 검정)
    shadow = obj.split()[3].filter(ImageFilter.GaussianBlur(6))
    sh = Image.new("RGBA", obj.size, (0, 0, 0, 0))
    sh.putalpha(shadow.point(lambda v: int(v * 0.30)))
    canvas.alpha_composite(sh, (x + 4, y + 5))
    canvas.alpha_composite(obj, (x, y))
    occupied.append(box)
    return box


def main() -> None:
    ap = make_parser("synthesize_multiobject")
    ap.add_argument("--per-class", type=int, default=800,
                    help="클래스당 생성 크롭 상한")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bgs = sorted(BACKGROUNDS.glob("*.jpg")) + sorted(BACKGROUNDS.glob("*.png"))
    if not bgs:
        sys.exit(f"배경 없음: {BACKGROUNDS}")
    print(f"배경 {len(bgs)}장, u2netp 컷아웃 풀 구축...")
    cutter = Cutter()
    pool = build_pool(cutter, rng)

    counts = {c: 0 for c in CLASSES}
    scene_id = 0
    while any(v < args.per_class for v in counts.values()):
        scene_id += 1
        if scene_id > args.per_class * 6:
            break
        # 클래스 조합: 60% 하드그룹 / 40% 전체 — 미달 클래스 우선
        group = (rng.choice(HARD_GROUPS) if rng.random() < 0.6 else CLASSES)
        cand = [c for c in group if pool[c]]
        cand.sort(key=lambda c: counts[c])
        k = rng.randint(2, 3)
        chosen = cand[:1] + rng.sample(cand[1:], min(k - 1, max(0, len(cand) - 1)))

        bg = Image.open(rng.choice(bgs)).convert("RGB").resize(CANVAS, Image.LANCZOS)
        canvas = bg.convert("RGBA")
        placed: list[tuple[str, tuple[int, int, int, int]]] = []
        occupied: list = []
        for cls in chosen:
            box = paste_object(canvas, rng.choice(pool[cls]), rng, occupied)
            if box is not None:
                placed.append((cls, box))
        if len(placed) < 2:
            continue

        scene = canvas.convert("RGB")
        for i, (cls, (x0, y0, x1, y1)) in enumerate(placed):
            if counts[cls] >= args.per_class:
                continue
            pw, ph = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
            box = (max(0, int(x0 - pw)), max(0, int(y0 - ph)),
                   min(CANVAS[0], int(x1 + pw)), min(CANVAS[1], int(y1 + ph)))
            crop = scene.crop(box)
            crop.thumbnail((STORE_MAX, STORE_MAX), Image.BILINEAR)
            out_dir = FINE_STAGING / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / f"synmo_{scene_id:05d}_{i}.jpg", "JPEG", quality=90)
            counts[cls] += 1

        if scene_id % 300 == 0:
            print(f"  scene {scene_id}: {counts}")

    total = sum(counts.values())
    print(f"\n생성 완료: 장면 {scene_id}, 크롭 {total:,}")
    for c, v in counts.items():
        print(f"  {c:16} {v:,}")
    print("SYNMO_DONE")


if __name__ == "__main__":
    main()
