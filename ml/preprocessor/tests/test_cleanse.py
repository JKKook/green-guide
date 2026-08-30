from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from greenguide_preprocessor import config
from greenguide_preprocessor.cleanse import cleanse, is_corrupt


def test_is_corrupt(tmp_path: Path, sample_image: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert is_corrupt(bad) is True
    assert is_corrupt(sample_image) is False


def test_cleanse_drops_corrupt_and_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    arr = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.png")
    Image.fromarray(arr).save(tmp_path / "dup.png")   # 동일 내용 → phash 중복
    (tmp_path / "bad.png").write_bytes(b"xx")
    items = [{"id": n, "label": "glass", "source_path": f"{n}.png", "filename": f"{n}.png"}
             for n in ("a", "dup", "bad")]
    catalog = {"dataset": "t", "classes": ["glass"], "count": 3, "items": items}
    cleaned, stats = cleanse(catalog)
    assert stats == {"input": 3, "corrupt": 1, "duplicate": 1, "kept": 1}
    assert [i["id"] for i in cleaned["items"]] == ["a"]
