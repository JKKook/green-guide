"""데이터 품질 보조 — 출처 키, 흐림·노출 지표.

- source_key: 파일명 접두어로 학습 아이템의 출처(AI-Hub 조건·Open Images 차수·TACO·
  사용자 피드백·합성·Kaggle)를 구분한다. 출처별 정확도·품질 통계의 공통 키.
- image_quality: 앱 `image_quality.dart` 와 같은 알고리즘(폭 256 다운스케일 → 그레이
  → 평균 밝기 + 3×3 Laplacian 분산)·같은 임계. 앱은 촬영 직후 재촬영을 안내하고,
  여기서는 인제스트 게이트와 데이터 리포트에 쓴다. 수치가 같은 잣대라야
  "학습 데이터의 흐림 비율"과 "사용자 사진의 흐림 비율"을 비교할 수 있다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

DARK_THRESHOLD = 45.0     # 평균 밝기(0~255) 이 미만이면 어두움 — 앱 _kDarkThreshold
BLUR_THRESHOLD = 80.0     # Laplacian 분산 이 미만이면 흐림 — 앱 _kBlurThreshold
ANALYSIS_WIDTH = 256      # 앱 _kAnalysisSize

_SOURCE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^aihub385_clean__aihub140_"), "aihub140"),
    (re.compile(r"^aihub385_(clean|multi|dirty|dirty_multi)__"), "aihub385_\\1"),
    (re.compile(r"^aihub"), "aihub"),
    (re.compile(r"^oi3_"), "oi3"),
    (re.compile(r"^oi2_"), "oi2"),
    (re.compile(r"^oi_"), "oi"),
    (re.compile(r"^taco_"), "taco"),
    (re.compile(r"^user_"), "user"),
    (re.compile(r"^realworld_"), "realworld"),
    (re.compile(r"^synmo_"), "synthetic"),
    (re.compile(r"^team_"), "team"),
)


def source_key(source_path: str) -> str:
    """preprocessor 루트 상대 경로 → 출처 키. 규칙에 없으면 'kaggle'(기본 세트)."""
    name = Path(source_path).name
    for pat, key in _SOURCE_RULES:
        m = pat.match(name)
        if m:
            return pat.sub(key, name[: m.end()]) if "\\1" in key else key
    return "kaggle"


def image_quality(img: Image.Image) -> tuple[float, float]:
    """(평균 밝기 0~255, Laplacian 분산). 앱과 같은 폭 256 다운스케일 뒤 계산."""
    w, h = img.size
    if w > ANALYSIS_WIDTH:
        img = img.resize((ANALYSIS_WIDTH, max(1, round(h * ANALYSIS_WIDTH / w))), Image.BILINEAR)
    g = np.asarray(img.convert("L"), dtype=np.float32)
    brightness = float(g.mean())
    if g.shape[0] < 3 or g.shape[1] < 3:
        return brightness, 0.0
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return brightness, float(lap.var())


def quality_issues(brightness: float, sharpness: float) -> list[str]:
    issues: list[str] = []
    if brightness < DARK_THRESHOLD:
        issues.append("too_dark")
    if sharpness < BLUR_THRESHOLD:
        issues.append("too_blurry")
    return issues
