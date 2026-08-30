from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from waste_common import imaging


def _sample(tmp_path: Path) -> Path:
    arr = (np.random.default_rng(0).random((600, 800, 3)) * 255).astype(np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")
    return path


def test_preprocess_matches_preprocessor_snapshot(tmp_path: Path) -> None:
    """waste-preprocessor Phase 0 characterization 과 동일한 출력 (EXIF 없는 이미지)."""
    arr = imaging.preprocess(_sample(tmp_path))
    assert arr.shape == (224, 224, 3) and arr.dtype == np.float32
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    assert digest == "9211f3e2a22e2c1571281c2389bae9a024adb6caca80c711aa69c46b8d1c975d"


def test_layouts_consistent(tmp_path: Path) -> None:
    p = _sample(tmp_path)
    hwc = imaging.preprocess(p, layout="hwc")
    chw = imaging.preprocess(p, layout="chw")
    nchw = imaging.preprocess(p, layout="nchw")
    assert chw.shape == (3, 224, 224) and nchw.shape == (1, 3, 224, 224)
    np.testing.assert_array_equal(imaging.hwc_to_chw(hwc), chw)
    np.testing.assert_array_equal(imaging.normalize_chw(imaging.hwc_to_chw(imaging.to_unit_array(
        imaging.resize_square(imaging.decode_rgb(p))))), chw)


def test_decode_handles_bytes_and_modes(tmp_path: Path) -> None:
    gray = tmp_path / "g.png"
    Image.new("L", (10, 20), 128).save(gray)
    img = imaging.decode_rgb(gray.read_bytes())
    assert img.mode == "RGB" and img.size == (10, 20)


def test_recompress_shrinks_only(tmp_path: Path) -> None:
    p = _sample(tmp_path)  # 800x600
    out = Image.open(__import__("io").BytesIO(imaging.recompress_for_storage(p, max_side=400)))
    assert out.size == (400, 300)
    small = Image.open(__import__("io").BytesIO(imaging.recompress_for_storage(p, max_side=2000)))
    assert small.size == (800, 600)


def test_content_type() -> None:
    assert imaging.content_type("a.JPG") == "image/jpeg"
    assert imaging.content_type(Path("a.xyz")) == "application/octet-stream"
