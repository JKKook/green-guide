"""inverse_freq_weights — 통합 전 _compute_class_weights/_capped_inverse_freq 출력 고정."""
from __future__ import annotations

import torch

from greenguide_classifier.train import inverse_freq_weights


def test_matches_legacy_flat_weights() -> None:
    # 13 클래스, count[i] = 3i+1 → 구 _compute_class_weights 출력
    counts = {i: 3 * i + 1 for i in range(13)}
    expected = torch.tensor([
        4.0, 4.0, 2.7142856121063232, 1.899999976158142, 1.4615384340286255,
        1.1875, 1.0, 0.8636363744735718, 0.7599999904632568, 0.6785714030265808,
        0.6129032373428345, 0.5588235259056091, 0.5135135054588318,
    ])
    torch.testing.assert_close(inverse_freq_weights(counts, 13, 4.0), expected)


def test_matches_legacy_hier_weights() -> None:
    # 빠진 클래스는 count=1 로 간주 — 구 _capped_inverse_freq 출력
    expected = torch.tensor([4.48, 0.224, 17.92, 3.2, 17.92])
    torch.testing.assert_close(inverse_freq_weights({0: 5, 1: 100, 3: 7}, 5, 4.0), expected)
