"""data_quality — 출처 키 규칙과 흐림·노출 지표(앱 image_quality.dart 와 같은 임계)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from greenguide_classifier.data_quality import (
    BLUR_THRESHOLD,
    DARK_THRESHOLD,
    image_quality,
    quality_issues,
    source_key,
)


def test_source_key_rules() -> None:
    cases = {
        "data/raw/fine-staging/electronics_boost/aihub385_clean__aihub140_이동전화단말기_20_X072.jpg": "aihub140",
        "data/raw/fine-staging/paper_boost/aihub385_dirty__TS_실내형분류기_B1_000003_0.jpg": "aihub385_dirty",
        "data/raw/fine-staging/electronics_boost/oi3_laptop_000123.jpg": "oi3",
        "data/raw/fine-staging/metal_boost/oi2_kettle_000001.jpg": "oi2",
        "data/raw/fine-staging/taco_vinyl/taco_000009.jpg": "taco",
        "data/raw/garbage-classification/electronics/user_ab12cd34.jpg": "user",
        "data/raw/synthetic_indoor/etc/synmo_000001.jpg": "synthetic",
        "data/raw/garbage-classification/paper/paper123.jpg": "kaggle",
    }
    for path, expected in cases.items():
        assert source_key(path) == expected, path


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")


def test_flat_gray_is_blurry_not_dark() -> None:
    b, s = image_quality(_img(np.full((300, 400), 128)))
    assert abs(b - 128) < 1
    assert s < BLUR_THRESHOLD
    assert quality_issues(b, s) == ["too_blurry"]


def test_dark_noise_is_dark_but_sharp() -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 60, size=(300, 400))
    b, s = image_quality(_img(arr))
    assert b < DARK_THRESHOLD
    assert s > BLUR_THRESHOLD
    assert quality_issues(b, s) == ["too_dark"]


def test_bright_noise_passes() -> None:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(300, 400))
    b, s = image_quality(_img(arr))
    assert quality_issues(b, s) == []


def test_downscale_matches_app_width() -> None:
    # 폭 1024 입력도 폭 256 기준으로 계산돼야 임계가 앱과 같은 잣대가 된다
    rng = np.random.default_rng(2)
    big = rng.integers(0, 256, size=(768, 1024))
    small = np.asarray(Image.fromarray(big.astype(np.uint8)).resize((256, 192), Image.BILINEAR))
    _, s_big = image_quality(_img(big))
    _, s_small = image_quality(_img(small))
    assert abs(s_big - s_small) / max(s_small, 1) < 0.05
