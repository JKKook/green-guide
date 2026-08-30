"""CAM-argmax 기반 다중재질 영역 추출 + 빗금(hatching) 오버레이.

per-class CAM (C, h, w) 에서 셀별 softmax→argmax 로 재질 지도를 만들고,
u2netp 객체 mask 안에서 confidence/최소 셀 수 필터로 "확실히 다른 재질"만 남김.
원본 이미지에 영역별 빗금 + (앱이 그릴) 라벨용 bbox 반환.
"""
from __future__ import annotations

import base64
import io
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw


# 필터 — "확실히 다른 재질만" (사용자 요구)
_CONF_THRESHOLD = 0.45    # 셀 클래스 확신 하한
_MIN_CELLS = 3            # 이 셀 수 미만 클래스는 노이즈로 간주, 제외
_MASK_THRESHOLD = 0.35    # 객체 점유 하한


def _softmax0(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=0, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=0, keepdims=True)


def _hex_to_rgb(hex_str: str | None, fallback=(120, 200, 120)) -> tuple[int, int, int]:
    if not hex_str:
        return fallback
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return fallback
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return fallback


def extract_regions(
    cam_all: np.ndarray,            # (C, h, w)
    mask_grid: np.ndarray,          # (h, w) 0~1 객체 점유
    labels: list[str],
    allowed_indices: list[int] | None = None,   # 재질 후보 제한 (Stage 1-3)
) -> list[dict]:
    """확실히 다른 재질 영역만 추출.

    Returns: [{class_index, slug, cells:[[r,c],...], bbox_norm, avg_conf}], 큰 영역 순.
    avg_conf 는 saliency(객체 점유) 가중 평균 (Stage 1-5) — 경계 셀 과대평가 억제.
    """
    if allowed_indices is not None:
        # 비후보 클래스는 셀 경쟁에서 제외 (−inf)
        masked = np.full_like(cam_all, -1e9)
        masked[allowed_indices] = cam_all[allowed_indices]
        cam_all = masked
    probs = _softmax0(cam_all)          # (C, h, w)
    cls = probs.argmax(axis=0)          # (h, w)
    conf = probs.max(axis=0)            # (h, w)
    h, w = cls.shape

    by_class: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if mask_grid[r, c] >= _MASK_THRESHOLD and conf[r, c] >= _CONF_THRESHOLD:
                # saliency 가중 확신 (Stage 1-5)
                by_class[int(cls[r, c])].append(
                    (r, c, float(conf[r, c] * mask_grid[r, c])))

    regions = []
    for ci, cells in by_class.items():
        if len(cells) < _MIN_CELLS:
            continue
        if ci >= len(labels):
            continue
        rs = [r for r, _, _ in cells]
        cs = [c for _, c, _ in cells]
        regions.append({
            "class_index": ci,
            "slug": labels[ci],
            "cells": [[r, c] for r, c, _ in cells],
            "bbox_norm": [
                min(cs) / w, min(rs) / h,
                (max(cs) + 1) / w, (max(rs) + 1) / h,
            ],
            "avg_conf": round(sum(cf for _, _, cf in cells) / len(cells), 3),
        })
    regions.sort(key=lambda x: -len(x["cells"]))
    return regions


def render_hatching(
    image_bytes: bytes,
    regions: list[dict],
    grid_h: int,
    grid_w: int,
    slug_to_color: dict[str, str | None],
    max_long_side: int = 720,
) -> str:
    """원본 이미지에 영역별 빗금(대각선) 오버레이 → base64 PNG (data URI).

    누끼 이미지를 따로 보여주지 않고 원본 위에 영역만 표시 (사용자 요구).
    """
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(orig.size) > max_long_side:
        scale = max_long_side / max(orig.size)
        orig = orig.resize(
            (int(orig.size[0] * scale), int(orig.size[1] * scale)), Image.BILINEAR,
        )
    w, h = orig.size
    ch, cw = h / grid_h, w / grid_w

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for region in regions:
        color = _hex_to_rgb(slug_to_color.get(region["slug"]))
        fill = color + (60,)     # 옅은 채움
        line = color + (200,)    # 빗금 선
        for (r, c) in region["cells"]:
            x0, y0 = c * cw, r * ch
            x1, y1 = x0 + cw, y0 + ch
            draw.rectangle([x0, y0, x1, y1], fill=fill,
                           outline=color + (160,), width=1)
            # 대각선 빗금 (셀 안에 2-3줄)
            step = max(8, int(cw / 4))
            xx = x0
            while xx < x1 + (y1 - y0):
                draw.line([(xx, y0), (xx - (y1 - y0), y1)], fill=line, width=1)
                xx += step

    composed = Image.alpha_composite(orig.convert("RGBA"), overlay)
    buf = io.BytesIO()
    composed.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
