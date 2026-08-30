"""벡터화 모듈 테스트."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from greenguide_preprocessor import config
from greenguide_preprocessor.vectorize import flatten, load_vector, save_vector


def test_flatten_produces_expected_dim() -> None:
    arr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype=np.float32)
    vec = flatten(arr)
    assert vec.shape == (config.VECTOR_DIM,)
    assert vec.dtype == np.float32


def test_flatten_rejects_wrong_shape() -> None:
    arr = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        flatten(arr)


def test_flatten_casts_to_float32() -> None:
    arr = np.zeros(
        (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype=np.float64,
    )
    vec = flatten(arr)
    assert vec.dtype == np.float32


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    # ImageNet 정규화 후 값 범위(-2.5 ~ 2.5)와 유사한 데이터로 검증
    rng = np.random.default_rng(0)
    vec = (rng.standard_normal(config.VECTOR_DIM, dtype=np.float32) * 0.5).astype(np.float32)
    path = save_vector(vec, item_id="roundtrip", base_dir=tmp_path)
    assert path.suffix == ".npz"
    loaded = load_vector("roundtrip", base_dir=tmp_path)
    assert loaded.dtype == np.float32
    # float16 다운캐스트로 인한 소폭 오차 허용
    np.testing.assert_allclose(vec, loaded, rtol=1e-2, atol=1e-3)


def test_save_uses_float16_internally(tmp_path: Path) -> None:
    vec = np.ones(config.VECTOR_DIM, dtype=np.float32)
    path = save_vector(vec, item_id="dtype_check", base_dir=tmp_path)
    with np.load(path, allow_pickle=False) as npz:
        assert npz["vector"].dtype == np.float16
