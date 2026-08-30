"""greenguide_classifier.infer 단위 테스트."""
from __future__ import annotations

import numpy as np
import torch

from greenguide_classifier.infer import pick_device, softmax


def test_softmax_1d_sums_to_one() -> None:
    p = softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert p.shape == (3,)
    np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-6)
    assert p.argmax() == 2


def test_softmax_axis_variants_match_reference() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)).astype(np.float32)
    ref_rows = np.exp(x - x.max(axis=1, keepdims=True))
    ref_rows /= ref_rows.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(softmax(x, axis=1), ref_rows, rtol=1e-6)
    np.testing.assert_allclose(softmax(x), ref_rows, rtol=1e-6)  # 2-D 에서 -1 == 1
    cam = rng.standard_normal((3, 2, 2)).astype(np.float32)
    np.testing.assert_allclose(softmax(cam, axis=0).sum(axis=0), np.ones((2, 2)), rtol=1e-6)


def test_softmax_large_values_stable() -> None:
    p = softmax(np.array([1000.0, 1000.0], dtype=np.float32))
    np.testing.assert_allclose(p, [0.5, 0.5])


def test_pick_device_returns_torch_device() -> None:
    d = pick_device()
    assert isinstance(d, torch.device)
    assert d.type in {"cuda", "mps", "cpu"}
