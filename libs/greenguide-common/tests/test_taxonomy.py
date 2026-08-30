"""taxonomy 무결성 (torch 불필요 부분). loss 관련 테스트는 greenguide-classifier/tests/test_hierarchy.py."""
from __future__ import annotations

import numpy as np

from greenguide_common.taxonomy import (
    COARSE_LABELS,
    COARSE_TO_INDEX,
    FINE_IDX_TO_COARSE_IDX,
    FINE_LABELS,
    FINE_TO_COARSE,
    LEGACY_LABEL_SUPERVISION,
    LEGACY_LABELS,
    NUM_COARSE,
    NUM_FINE,
    STAGING_DIR_SUPERVISION,
    TAXONOMY,
    rollup_fine_probs,
    same_guidance,
    supervision_index,
)


def test_partition() -> None:
    seen: set[str] = set()
    for coarse, children in TAXONOMY.items():
        assert children, coarse
        for f in children:
            assert f not in seen, f
            seen.add(f)
    assert seen == set(FINE_LABELS)
    assert (len(FINE_LABELS), len(COARSE_LABELS)) == (NUM_FINE, NUM_COARSE)
    for f in FINE_LABELS:
        assert f in TAXONOMY[FINE_TO_COARSE[f]]
    for fi, ci in enumerate(FINE_IDX_TO_COARSE_IDX):
        assert COARSE_LABELS[ci] == FINE_TO_COARSE[FINE_LABELS[fi]]


def test_supervision_maps_valid() -> None:
    for mapping in (LEGACY_LABEL_SUPERVISION, STAGING_DIR_SUPERVISION):
        for _, (kind, slug) in mapping.items():
            space = FINE_LABELS if kind == "fine" else COARSE_LABELS
            assert space[supervision_index(kind, slug)] == slug
    assert set(LEGACY_LABELS) <= set(LEGACY_LABEL_SUPERVISION)


def test_rollup_numpy() -> None:
    rng = np.random.default_rng(0)
    p = rng.random((4, NUM_FINE)); p /= p.sum(1, keepdims=True)
    rolled = rollup_fine_probs(p)
    assert rolled.shape == (4, NUM_COARSE)
    np.testing.assert_allclose(rolled.sum(1), 1.0)
    gi = COARSE_TO_INDEX["glass"]
    np.testing.assert_allclose(rolled[:, gi], p[:, [i for i, c in enumerate(FINE_IDX_TO_COARSE_IDX) if c == gi]].sum(1))


def test_same_guidance() -> None:
    assert same_guidance("carton", "paper_cup") and same_guidance("pet", "pet")
    assert not same_guidance("glass_clear", "glass_deposit")
