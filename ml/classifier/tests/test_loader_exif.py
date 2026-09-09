"""_load_rgb_chw01 의 EXIF 방향 보정 — 서빙(decode_rgb)과 동일 픽셀 검증."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from greenguide_classifier import config
from greenguide_classifier.dataset import _load_rgb_chw01


def _asym_image() -> Image.Image:
    """위쪽 1/3 빨강, 나머지 파랑 — 회전하면 픽셀 분포가 달라지는 이미지."""
    arr = np.zeros((90, 60, 3), dtype=np.uint8)
    arr[:30, :, 0] = 255
    arr[30:, :, 2] = 255
    return Image.fromarray(arr)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_loader_applies_exif_orientation(tmp_path, monkeypatch, orientation) -> None:
    img = _asym_image()
    exif = Image.Exif()
    exif[0x0112] = orientation
    path = tmp_path / f"o{orientation}.jpg"
    img.save(path, exif=exif.tobytes())

    monkeypatch.setattr(config, "PREPROCESSOR_ROOT", tmp_path)
    x = _load_rgb_chw01({"id": "t", "source_path": path.name})
    assert x.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    # 기대값: exif_transpose 로 세운 이미지를 같은 방식으로 resize
    from PIL import ImageOps
    with Image.open(path) as im:
        up = ImageOps.exif_transpose(im).convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        expected = np.asarray(up, dtype=np.float32) / 255.0
    np.testing.assert_allclose(x.numpy().transpose(1, 2, 0), expected, atol=1e-6)

    # orientation≠1 이면 '보정 없이' 로드한 것과는 달라야 검증이 유의미
    with Image.open(path) as im:
        raw = np.asarray(
            im.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR),
            dtype=np.float32) / 255.0
    if orientation != 1:
        assert not np.allclose(x.numpy().transpose(1, 2, 0), raw)
