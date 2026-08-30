#!/usr/bin/env python3
"""실내 배경 합성 — 실사용 도메인 갭(책상·바닥·실내조명) 대응 학습 데이터 생성.

배경: AI-Hub 크롭(선별장 도메인)으로 학습한 모델이 실내 생활공간 사진에 약함.
실사용 사진의 저saliency 영역에서 배경 패치를 수확하고, fine-staging 크롭을
u2netp 누끼로 얹어 "실내에 놓인 폐기물" 분포를 합성한다 (SEMANTIC_FUSION 백로그).

출력: synth_indoor_staging/<staging_label>/synmo_in_*.jpg
- fine-staging 에 직접 쓰지 않음 — **사이클 도중 fine-staging 수정 금지** 규율.
  v7 완료 후 integrate 단계에서 병합 + splits rebuild (v8).
- 파일명 synmo_ prefix → hier_dataset._is_synthetic 이 frozen test 에서 자동 제외.

실행: .venv/bin/python scripts/synthesize_indoor_scenes.py [--per-class 300]
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image, ImageFilter, ImageOps  # noqa: E402

FINE_STAGING = Path("/Users/ethan/practice/waste/waste-preprocessor/data/raw/fine-staging")
REALWORLD_DIR = Path(
    "/private/tmp/claude-501/-Users-ethan-practice-waste/"
    "142cc691-4ab6-48ea-a632-274f14f81459/scratchpad/realworld")
U2NETP = Path("/Users/ethan/practice/waste/waste-api/models/u2netp.onnx")
OUT_DIR = Path("/Users/ethan/practice/waste/synth_indoor_staging")

CANVAS = 640                     # 합성 캔버스 (긴 변)
BG_MAX_SALIENCY = 0.10           # 배경 패치 평균 saliency 상한
_U2_MEAN = (0.485, 0.456, 0.406)
_U2_STD = (0.229, 0.224, 0.225)


class Saliency:
    def __init__(self) -> None:
        self.sess = ort.InferenceSession(str(U2NETP), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def mask(self, img: Image.Image) -> np.ndarray:
        """RGB 이미지 → (320,320) saliency 0~1."""
        im = img.convert("RGB").resize((320, 320), Image.LANCZOS)
        a = np.asarray(im).astype(np.float64)
        mx = a.max()
        if mx > 0:
            a = a / mx
        for c in range(3):
            a[:, :, c] = (a[:, :, c] - _U2_MEAN[c]) / _U2_STD[c]
        x = a.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0, 0]
        mi, ma = float(out.min()), float(out.max())
        return (out - mi) / (ma - mi + 1e-8)


def harvest_backgrounds(sal: Saliency, rng: random.Random) -> list[Image.Image]:
    """실사용 사진의 저saliency 영역에서 배경 패치 수확 (+증강)."""
    patches: list[Image.Image] = []
    for p in sorted(REALWORLD_DIR.glob("*.jpg")):
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        m = sal.mask(img)
        W, H = img.size
        for _ in range(12):  # 사진당 최대 12회 시도
            side = rng.randint(min(W, H) // 3, min(W, H) * 2 // 3)
            x0 = rng.randint(0, W - side)
            y0 = rng.randint(0, H - side)
            gx0, gy0 = int(x0 / W * 320), int(y0 / H * 320)
            gs = max(1, int(side / W * 320)), max(1, int(side / H * 320))
            region = m[gy0:gy0 + gs[1], gx0:gx0 + gs[0]]
            if region.size and float(region.mean()) < BG_MAX_SALIENCY:
                patches.append(img.crop((x0, y0, x0 + side, y0 + side))
                               .resize((CANVAS, CANVAS), Image.LANCZOS))
    print(f"[bg] 배경 패치 {len(patches)}개 수확 (실사용 {len(list(REALWORLD_DIR.glob('*.jpg')))}장)")
    return patches


def augment_bg(bg: Image.Image, rng: random.Random) -> Image.Image:
    out = bg.copy()
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        out = out.rotate(rng.choice((90, 180, 270)))
    b = 0.75 + rng.random() * 0.5
    out = Image.eval(out, lambda v: min(255, int(v * b)))
    return out


def paste_object(bg: Image.Image, obj: Image.Image, alpha: np.ndarray,
                 rng: random.Random) -> Image.Image:
    """u2 누끼 alpha 로 객체를 배경 위 무작위 위치·크기에 합성."""
    canvas = bg.copy()
    # 객체 목표 크기: 배경 변의 30~70%
    target = int(CANVAS * (0.35 + rng.random() * 0.40))
    scale = target / max(obj.size)
    ow, oh = max(24, int(obj.size[0] * scale)), max(24, int(obj.size[1] * scale))
    obj_r = obj.resize((ow, oh), Image.LANCZOS)
    a_img = Image.fromarray((alpha * 255).astype(np.uint8), "L").resize(
        (ow, oh), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.5))
    deg = rng.uniform(-15, 15)
    obj_r = obj_r.rotate(deg, expand=True, resample=Image.BICUBIC)
    a_img = a_img.rotate(deg, expand=True, resample=Image.BICUBIC)
    px = rng.randint(0, max(1, CANVAS - obj_r.size[0]))
    py = rng.randint(0, max(1, CANVAS - obj_r.size[1]))
    canvas.paste(obj_r, (px, py), a_img)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sal = Saliency()
    bgs = harvest_backgrounds(sal, rng)
    if len(bgs) < 10:
        raise SystemExit("배경 패치 부족 — BG_MAX_SALIENCY 완화 필요")

    label_dirs = sorted(d for d in FINE_STAGING.iterdir() if d.is_dir())
    for d in label_dirs:
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        if not files:
            continue
        out = OUT_DIR / d.name
        out.mkdir(parents=True, exist_ok=True)
        made = skipped = 0
        rng.shuffle(files)
        for i, f in enumerate(files):
            if made >= args.per_class:
                break
            try:
                obj = Image.open(f).convert("RGB")
            except OSError:
                continue
            m = sal.mask(obj)
            # 객체 누끼 품질 가드 — saliency 가 흐릿하면(객체 경계 불명) 스킵
            if float((m > 0.5).mean()) < 0.12:
                skipped += 1
                continue
            alpha = np.asarray(
                Image.fromarray((m * 255).astype(np.uint8)).resize(obj.size, Image.BILINEAR)
            ).astype(np.float32) / 255.0
            comp = paste_object(augment_bg(rng.choice(bgs), rng), obj, alpha, rng)
            comp.save(out / f"synmo_in_{d.name}_{made:05d}.jpg", quality=90)
            made += 1
        print(f"[synth] {d.name}: {made}장 생성 (누끼불량 스킵 {skipped})")

    total = sum(1 for _ in OUT_DIR.rglob("synmo_in_*.jpg"))
    print(f"[synth] 총 {total:,}장 → {OUT_DIR} (v7 완료 후 integrate + splits rebuild)")


if __name__ == "__main__":
    main()
