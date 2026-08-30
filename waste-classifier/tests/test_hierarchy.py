"""계층 taxonomy·loss 무결성 테스트."""
from __future__ import annotations

import math

import torch

from src.taxonomy import (
    COARSE_LABELS, COARSE_TO_INDEX, FINE_IDX_TO_COARSE_IDX, FINE_LABELS,
    FINE_TO_COARSE, LEGACY_LABEL_SUPERVISION, NUM_COARSE, NUM_FINE,
    STAGING_DIR_SUPERVISION, TAXONOMY, rollup_fine_probs, supervision_index,
)


def test_taxonomy_partition():
    """모든 fine 은 정확히 하나의 coarse 에 속하고, 중복이 없다."""
    seen = set()
    for coarse, children in TAXONOMY.items():
        assert children, f"{coarse} 에 children 없음"
        for f in children:
            assert f not in seen, f"fine {f} 이 두 coarse 에 중복"
            seen.add(f)
    assert seen == set(FINE_LABELS)
    assert len(FINE_LABELS) == NUM_FINE
    assert len(COARSE_LABELS) == NUM_COARSE


def test_fine_to_coarse_consistency():
    for f in FINE_LABELS:
        c = FINE_TO_COARSE[f]
        assert f in TAXONOMY[c]
    for fi, ci in enumerate(FINE_IDX_TO_COARSE_IDX):
        assert COARSE_LABELS[ci] == FINE_TO_COARSE[FINE_LABELS[fi]]


def test_legacy_supervision_valid():
    """legacy 매핑의 대상 slug 가 실제 공간에 존재."""
    for label, (kind, slug) in LEGACY_LABEL_SUPERVISION.items():
        idx = supervision_index(kind, slug)
        space = FINE_LABELS if kind == "fine" else COARSE_LABELS
        assert space[idx] == slug, f"legacy {label} → {kind}:{slug} 불일치"


def test_staging_supervision_valid():
    for d, (kind, slug) in STAGING_DIR_SUPERVISION.items():
        idx = supervision_index(kind, slug)
        space = FINE_LABELS if kind == "fine" else COARSE_LABELS
        assert space[idx] == slug, f"staging {d} → {kind}:{slug} 불일치"


def test_rollup_probs_sum():
    """롤업 후에도 확률 합=1, children 합산 정확성."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, NUM_FINE), dim=1)
    rolled = rollup_fine_probs(probs)
    assert rolled.shape == (4, NUM_COARSE)
    assert torch.allclose(rolled.sum(dim=1), torch.ones(4), atol=1e-5)
    # glass 롤업 = 5개 children 합
    gi = COARSE_TO_INDEX["glass"]
    children_idx = [i for i, c in enumerate(FINE_IDX_TO_COARSE_IDX) if c == gi]
    assert len(children_idx) == 5
    assert torch.allclose(rolled[:, gi], probs[:, children_idx].sum(dim=1), atol=1e-6)


def test_hierarchical_loss_math():
    """롤업 NLL 이 수학적으로 -log(Σ P(children)) 과 일치."""
    from src.hier_train import HierarchicalLoss

    crit = HierarchicalLoss(torch.ones(NUM_FINE), torch.ones(NUM_COARSE))
    torch.manual_seed(1)
    logits = torch.randn(3, NUM_FINE)

    # coarse 감독: glass
    gi = COARSE_TO_INDEX["glass"]
    is_fine = torch.zeros(3, dtype=torch.long)
    sup = torch.full((3,), gi, dtype=torch.long)
    loss = crit(logits, is_fine, sup)

    probs = torch.softmax(logits, dim=1)
    p_glass = rollup_fine_probs(probs)[:, gi]
    expected = (-p_glass.log()).mean()
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-4)


def test_hierarchical_loss_fine_matches_ce():
    """fine 감독만 있을 때 표준 CE 와 동일."""
    import torch.nn.functional as F
    from src.hier_train import HierarchicalLoss

    crit = HierarchicalLoss(torch.ones(NUM_FINE), torch.ones(NUM_COARSE))
    torch.manual_seed(2)
    logits = torch.randn(5, NUM_FINE)
    y = torch.randint(0, NUM_FINE, (5,))
    loss = crit(logits, torch.ones(5, dtype=torch.long), y)
    assert math.isclose(loss.item(), F.cross_entropy(logits, y).item(), rel_tol=1e-5)


def test_blueprint_key_classes_present():
    """청사진 핵심 신규 클래스가 taxonomy 에 존재."""
    for fine in ("battery", "carton", "paper_cup", "glass_deposit",
                 "light_bulb", "vinyl_dirty", "styrofoam_dirty", "pet"):
        assert fine in FINE_LABELS
    for coarse in ("hazardous", "paper_pack", "trash"):
        assert coarse in COARSE_LABELS
