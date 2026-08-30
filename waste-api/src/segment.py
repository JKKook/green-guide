"""객체 누끼(saliency segmentation) — u2netp ONNX 직접 사용.

rembg 라이브러리(numpy 2.x·scikit-image·numba 등 무거운 의존성) 없이
u2netp.onnx(4.4MB) 만 기존 onnxruntime 으로 구동. rembg 의 u2net 전/후처리 재현.

용도: 캡처 이미지에서 주요 객체의 mask + bbox 추출 → 앱이 배경 dim + 라벨 오버레이.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.core import config
from src.core.singleton import lazy_singleton


_U2NET_MEAN = (0.485, 0.456, 0.406)
_U2NET_STD = (0.229, 0.224, 0.225)
_SIZE = 320
_MASK_THRESHOLD = 64   # 0-255, bbox 추출 시 객체로 간주할 alpha 하한


def _model_path() -> Path:
    bundled = config.PROJECT_ROOT / "models" / "u2netp.onnx"
    return bundled


class Segmenter:
    """u2netp saliency 모델 wrapper."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or _model_path()
        self.available = self.model_path.exists()
        self.session: ort.InferenceSession | None = None
        if self.available:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        im = img.convert("RGB").resize((_SIZE, _SIZE), Image.LANCZOS)
        ary = np.array(im).astype(np.float64)
        mx = ary.max()
        if mx > 0:
            ary = ary / mx
        tmp = np.zeros((_SIZE, _SIZE, 3), dtype=np.float64)
        for c in range(3):
            tmp[:, :, c] = (ary[:, :, c] - _U2NET_MEAN[c]) / _U2NET_STD[c]
        chw = tmp.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return chw

    def object_mask_grid(self, image_bytes: bytes, grid: int) -> np.ndarray:
        """u2netp saliency → grid×grid 객체 점유 비율 (0~1). 없으면 전부 1."""
        if not self.available or self.session is None:
            return np.ones((grid, grid), dtype=np.float32)
        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inp = self._preprocess(orig)
        out = self.session.run(None, {self.input_name: inp})[0][0, 0]
        mi, ma = float(out.min()), float(out.max())
        out = (out - mi) / (ma - mi + 1e-8)
        m = Image.fromarray((out * 255).astype(np.uint8)).resize(
            (grid, grid), Image.BILINEAR,
        )
        return np.array(m).astype(np.float32) / 255.0

    def segment(self, image_bytes: bytes) -> dict:
        """이미지 → {cutout_base64, bbox_norm, object_ratio}.

        - cutout_base64: 객체만 남기고 배경을 투명하게 한 RGBA PNG (data URI),
          긴 변 최대 512. 앱이 [dim 원본] 위에 이 cutout 을 겹쳐 객체를 부각.
        - bbox_norm: [x0, y0, x1, y1] 0~1 정규화 (라벨 위치용, 없으면 None).
        - object_ratio: 객체가 차지하는 면적 비율.
        """
        if not self.available or self.session is None:
            return {"cutout_base64": None, "bbox_norm": None, "object_ratio": 0.0}

        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ow, oh = orig.size

        inp = self._preprocess(orig)
        out = self.session.run(None, {self.input_name: inp})[0]  # (1,1,320,320)
        pred = out[0, 0, :, :]
        mi, ma = float(pred.min()), float(pred.max())
        if ma - mi > 1e-8:
            pred = (pred - mi) / (ma - mi)
        else:
            pred = np.zeros_like(pred)

        # 응답 크기 위해 긴 변 512 제한
        long_side = max(ow, oh)
        scale = min(1.0, 512 / long_side)
        out_w, out_h = int(ow * scale), int(oh * scale)

        alpha = Image.fromarray((pred * 255).astype(np.uint8), mode="L").resize(
            (out_w, out_h), Image.LANCZOS,
        )
        rgb = orig.resize((out_w, out_h), Image.LANCZOS)

        # cutout — RGBA (배경 alpha=saliency)
        cutout = rgb.convert("RGBA")
        cutout.putalpha(alpha)

        # bbox (정규화)
        m = np.array(alpha)
        ys, xs = np.where(m > _MASK_THRESHOLD)
        bbox_norm = None
        object_ratio = 0.0
        if len(xs) > 0:
            bbox_norm = [
                float(xs.min() / out_w), float(ys.min() / out_h),
                float(xs.max() / out_w), float(ys.max() / out_h),
            ]
            object_ratio = float((m > _MASK_THRESHOLD).mean())

        buf = io.BytesIO()
        cutout.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "cutout_base64": f"data:image/png;base64,{b64}",
            "bbox_norm": bbox_norm,
            "object_ratio": round(object_ratio, 4),
        }


@lazy_singleton
def get_segmenter() -> Segmenter:
    return Segmenter()


# 병합 성분 재분리 — u2netp 는 인접 객체 사이 후광까지 salient 라 기본
# 임계(0.4)에선 한 성분으로 붙는다. 두 전략으로 코어(씨앗)를 찾는다:
#  - 임계 상승: 객체 중심부는 saliency 가 높고 경계 골짜기는 낮음
#  - 침식: 골짜기가 얕아 임계로 안 갈라져도, 물체 사이 좁은 다리는 침식에 끊김
#    (EXIF 회전 적용본 실측 — 임계만으론 붙은 병뚜껑이 0.88에야 2개, 침식 16회면 3개)
_SPLIT_LEVELS = (0.55, 0.7, 0.8, 0.88)
_ERODE_STEPS = (4, 8, 12, 16)
_SEED_MIN_FRAC = 0.04    # 코어 유효 최소변 — 노이즈 조각 배제
_SEED_MAX_N = 6          # 코어가 이보다 많으면 텍스처(키보드 키 등)로 보고 분리 포기


def _erode_once(b: np.ndarray) -> np.ndarray:
    """4-이웃 이진 침식 1회 (numpy 시프트 — scipy 불요)."""
    e = b.copy()
    e[1:, :] &= b[:-1, :]
    e[:-1, :] &= b[1:, :]
    e[:, 1:] &= b[:, :-1]
    e[:, :-1] &= b[:, 1:]
    return e


def _saliency_prob(image_bytes: bytes) -> np.ndarray | None:
    """u2netp saliency → (320,320) 0~1 확률. 모델 없음/평탄 시 None."""
    seg = get_segmenter()
    if not seg.available or seg.session is None:
        return None
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out = seg.session.run(None, {seg.input_name: seg._preprocess(orig)})[0][0, 0]
    mi, ma = float(out.min()), float(out.max())
    if ma - mi < 1e-8:
        return None
    return (out - mi) / (ma - mi)


def _bfs_components(
    binary: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    """4-연결 성분 — (면적, bbox_px(x0,y0,x1,y1), mask) 목록, 면적 내림차순."""
    from collections import deque
    visited = np.zeros_like(binary, dtype=bool)
    comps: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for sy, sx in zip(*np.nonzero(binary)):
        if visited[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        mask = np.zeros_like(binary, dtype=bool)
        mask[sy, sx] = True
        x0 = x1 = int(sx)
        y0 = y1 = int(sy)
        area = 0
        while q:
            cy, cx = q.popleft()
            area += 1
            x0, x1 = min(x0, cx), max(x1, cx)
            y0, y1 = min(y0, cy), max(y1, cy)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < _SIZE and 0 <= nx < _SIZE
                        and binary[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    mask[ny, nx] = True
                    q.append((ny, nx))
        comps.append((area, (x0, y0, x1, y1), mask))
    comps.sort(key=lambda t: -t[0])
    return comps


def _partition_merged(
    prob: np.ndarray, mask: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray] | None:
    """병합 의심 성분을 상위 임계 코어로 재분리 (watershed-lite).

    1) 임계 상승(_SPLIT_LEVELS)·침식(_ERODE_STEPS) 전 전략에서 성분 내부 코어를
       찾아, 유효 코어(≥_SEED_MIN_FRAC)가 가장 많이 갈라지는 결과 선택
       (2개 미만이면 분리 실패 → None)
    2) 코어들을 씨앗으로 multi-source BFS — 성분의 모든 픽셀을 가장 가까운
       코어에 귀속시켜 분할. bbox 가 코어 크기로 쪼그라들지 않고 원 성분
       범위를 나눠 갖는다.
    반환: (라벨별 bbox_px 목록, 라벨맵(-1=성분 밖)) — 호출부가 탭 지점 라벨 조회 가능.
    """
    from collections import deque
    seed_min = _SIZE * _SEED_MIN_FRAC

    def _valid_cores(binary: np.ndarray) -> list[np.ndarray]:
        return [
            m for _, (cx0, cy0, cx1, cy1), m in _bfs_components(binary)
            if (cx1 - cx0) >= seed_min and (cy1 - cy0) >= seed_min
        ]

    best: list[np.ndarray] | None = None
    for t in _SPLIT_LEVELS:
        cores = _valid_cores((prob > t) & mask)
        if 2 <= len(cores) <= _SEED_MAX_N and (best is None or len(cores) > len(best)):
            best = cores
    eroded = mask
    for k in range(1, max(_ERODE_STEPS) + 1):
        eroded = _erode_once(eroded)
        if k not in _ERODE_STEPS:
            continue
        cores = _valid_cores(eroded)
        # 동수면 침식 쪽 우선(≥) — 씨앗이 물체 중심으로 더 수렴해 분할이 안정적
        if 2 <= len(cores) <= _SEED_MAX_N and (best is None or len(cores) >= len(best)):
            best = cores
    if best is None:
        return None

    labels = np.full(mask.shape, -1, dtype=np.int16)
    q = deque()
    for li, core in enumerate(best):
        for cy, cx in zip(*np.nonzero(core)):
            labels[cy, cx] = li
            q.append((int(cy), int(cx)))
    while q:
        cy, cx = q.popleft()
        li = labels[cy, cx]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < _SIZE and 0 <= nx < _SIZE
                    and mask[ny, nx] and labels[ny, nx] < 0):
                labels[ny, nx] = li
                q.append((ny, nx))

    # 조각 면적 하한 — 케이블·그림자 등 가늘고 성긴 노이즈 조각 배제
    min_area = _SIZE * _SIZE * 0.015
    bboxes: list[tuple[int, int, int, int]] = []
    for li in range(len(best)):
        ys, xs = np.nonzero(labels == li)
        if len(xs) < min_area:
            continue
        bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    if len(bboxes) < 2:
        return None   # 유효 조각이 2개 미만이면 분리 의미 없음 — 원 성분 유지
    return bboxes, labels


def component_bbox_at(
    image_bytes: bytes, tap_x: float, tap_y: float,
    threshold: float = 0.4, search_radius_frac: float = 0.08,
) -> list[float] | None:
    """탭 지점(정규화 0~1)이 속한 saliency 객체의 bbox_norm 반환.

    탭-투-셀렉트용: 혼재 장면에서 사용자가 지목한 객체만 분리.
    1) u2netp 마스크(320²) → threshold 이진화
    2) 탭 지점이 배경이면 주변 반경에서 가장 가까운 객체 픽셀 탐색
    3) BFS flood-fill 로 연결 성분 추출 → 병합 의심 시 상위 임계 재분리 후
       탭 지점이 속한 분할만 반환 (인접 객체 혼입 방지)
    실패(마스크 없음/성분 없음) 시 None — 호출부가 window-crop fallback.
    """
    prob = _saliency_prob(image_bytes)
    if prob is None:
        return None
    binary = prob > threshold

    tx = min(max(int(tap_x * _SIZE), 0), _SIZE - 1)
    ty = min(max(int(tap_y * _SIZE), 0), _SIZE - 1)

    # 탭 지점이 배경이면 반경 내 최근접 객체 픽셀로 스냅
    if not binary[ty, tx]:
        r = max(1, int(_SIZE * search_radius_frac))
        ys, xs = np.where(
            binary[max(0, ty - r):ty + r + 1, max(0, tx - r):tx + r + 1])
        if len(xs) == 0:
            return None
        d2 = (ys - min(ty, r)) ** 2 + (xs - min(tx, r)) ** 2
        k = int(d2.argmin())
        ty = max(0, ty - r) + int(ys[k])
        tx = max(0, tx - r) + int(xs[k])

    # BFS flood fill (scipy 없이 — 320² 는 충분히 가벼움)
    from collections import deque
    mask = np.zeros_like(binary, dtype=bool)
    q = deque([(ty, tx)])
    mask[ty, tx] = True
    x0, y0, x1, y1 = tx, ty, tx, ty
    while q:
        cy, cx = q.popleft()
        x0, x1 = min(x0, cx), max(x1, cx)
        y0, y1 = min(y0, cy), max(y1, cy)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < _SIZE and 0 <= nx < _SIZE
                    and binary[ny, nx] and not mask[ny, nx]):
                mask[ny, nx] = True
                q.append((ny, nx))

    # 병합 의심 → 탭 지점이 속한 분할만 (인접 객체가 크롭에 섞이는 것 방지)
    part = _partition_merged(prob, mask)
    if part is not None:
        bboxes, labels = part
        li = int(labels[ty, tx])
        if 0 <= li < len(bboxes):
            x0, y0, x1, y1 = bboxes[li]

    # 너무 작은 성분(노이즈)은 무시
    if (x1 - x0) < _SIZE * 0.03 or (y1 - y0) < _SIZE * 0.03:
        return None
    return [x0 / _SIZE, y0 / _SIZE, (x1 + 1) / _SIZE, (y1 + 1) / _SIZE]


def all_component_bboxes(
    image_bytes: bytes, threshold: float = 0.4,
    min_side_frac: float = 0.06, max_n: int = 5,
) -> list[list[float]]:
    """u2netp saliency 객체 후보 bbox_norm 목록 (면적 내림차순, 최대 max_n).

    탐지-후-분류용: 혼재 장면의 각 객체 후보를 분리한다.
    saliency 는 인스턴스 세그가 아니라 인접 객체가 한 성분으로 붙으므로,
    성분마다 상위 임계 재분리(_partition_merged)를 시도해 나눈다.
    (실사용 51장 실측: 재분리 전 후보 2+ 비율 3/51 — 다중 라벨링 사실상 미발동)
    """
    prob = _saliency_prob(image_bytes)
    if prob is None:
        return []
    min_side = _SIZE * min_side_frac

    out: list[tuple[int, list[float]]] = []
    for area, bbox, mask in _bfs_components(prob > threshold):
        part = _partition_merged(prob, mask)
        parts = part[0] if part is not None else [bbox]
        valid = [
            (px0, py0, px1, py1) for (px0, py0, px1, py1) in parts
            if (px1 - px0) >= min_side and (py1 - py0) >= min_side
        ]
        if not valid and (bbox[2] - bbox[0]) >= min_side and (bbox[3] - bbox[1]) >= min_side:
            valid = [bbox]   # 분할 조각이 전부 미달이면 원 성분으로 폴백
        for (px0, py0, px1, py1) in valid:
            out.append((
                (px1 - px0) * (py1 - py0),
                [px0 / _SIZE, py0 / _SIZE,
                 (px1 + 1) / _SIZE, (py1 + 1) / _SIZE],
            ))

    out.sort(key=lambda t: -t[0])
    return [bb for _, bb in out[:max_n]]


def grabcut_object_at(
    image_bytes: bytes, tap_x: float, tap_y: float, grid: int,
    max_side: int = 480,
) -> tuple[np.ndarray | None, list[float] | None]:
    """탭 지점 물건의 GrabCut 전경 실루엣 → (grid×grid 점유, bbox_norm).

    u2netp saliency 는 '시선 지도'라 책상 경계·이웃 물체까지 밝아져 탭 실루엣이
    번지고(빗금 어긋남 리포트), 맞닿은 물체는 성분이 붙어버린다. GrabCut 은
    탭 중심 창을 전경 후보로 색 모델을 학습해 물건 경계를 픽셀 수준으로 분리.
    실패(전경 없음·탭이 배경 판정) 시 (None, None) — 호출부 saliency fallback.
    """
    import cv2  # noqa: PLC0415

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    arr = cv2.cvtColor(
        np.array(img.resize((sw, sh), Image.BILINEAR)), cv2.COLOR_RGB2BGR)

    half = int(0.35 * min(sw, sh))
    cx, cy = int(tap_x * sw), int(tap_y * sh)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(sw, cx + half), min(sh, cy + half)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, None
    mask = np.zeros((sh, sw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(arr, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd,
                    4, cv2.GC_INIT_WITH_RECT)
    except Exception:  # noqa: BLE001
        return None, None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    if fg.sum() == 0:
        return None, None

    # 탭 픽셀 포함 연결 성분만 (이웃 물체 유입 차단)
    _, lab = cv2.connectedComponents(fg)
    ty, tx = min(sh - 1, cy), min(sw - 1, cx)
    tl = lab[ty, tx]
    if tl == 0:
        ys, xs = np.nonzero(fg)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        i = int(d2.argmin())
        if d2[i] > (0.06 * min(sw, sh)) ** 2:
            return None, None  # 탭 근방에 전경 없음
        tl = lab[ys[i], xs[i]]
    comp = (lab == tl).astype(np.float32)
    ys, xs = np.nonzero(comp)
    bbox = [float(xs.min()) / sw, float(ys.min()) / sh,
            float(xs.max() + 1) / sw, float(ys.max() + 1) / sh]
    grid_occ = cv2.resize(comp, (grid, grid), interpolation=cv2.INTER_AREA)
    return grid_occ.astype(np.float32), bbox
