"""저장용 재압축 — WebP 640px q75 계약."""
import io

from PIL import Image

from src.uploads import _recompress_for_storage


def _jpeg(w: int, h: int, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.effect_noise((w, h), 64).convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def test_large_photo_becomes_webp_640():
    src = _jpeg(1600, 1200)
    out, ct = _recompress_for_storage(src, "image/jpeg")
    assert ct == "image/webp"
    im = Image.open(io.BytesIO(out))
    assert im.format == "WEBP"
    assert max(im.size) == 640
    assert len(out) < len(src) // 4  # 노이즈 이미지도 4배 이상 절감


def test_exif_orientation_is_baked():
    """EXIF 회전(6=90°CW)을 픽셀에 반영 — 저장본은 EXIF 없이도 바로 서야 한다."""
    im = Image.effect_noise((800, 400), 64).convert("RGB")
    exif = Image.Exif(); exif[0x0112] = 6
    buf = io.BytesIO(); im.save(buf, "JPEG", exif=exif.tobytes())
    out, _ = _recompress_for_storage(buf.getvalue(), "image/jpeg")
    w, h = Image.open(io.BytesIO(out)).size
    assert h > w  # 가로 사진이 회전 적용되어 세로로


def test_tiny_input_kept_as_is():
    src = _jpeg(120, 90, quality=30)
    out, ct = _recompress_for_storage(src, "image/jpeg")
    if ct == "image/jpeg":
        assert out == src
    else:
        assert len(out) < len(src)


def test_garbage_bytes_fail_open():
    out, ct = _recompress_for_storage(b"not an image", "image/jpeg")
    assert (out, ct) == (b"not an image", "image/jpeg")
