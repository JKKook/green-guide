"""Characterization test — taxonomy 매핑 스냅샷 고정 (Phase 0).

taxonomy 를 greenguide_common 으로 이동해도 fine/coarse 순서·매핑이 1비트도 바뀌지 않아야 한다.
의도적으로 클래스를 추가/변경할 때만 `tests/snapshots/taxonomy.json` 을 갱신한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from greenguide_classifier import taxonomy

SNAPSHOT = Path(__file__).parent / "snapshots" / "taxonomy.json"


def current_snapshot() -> dict:
    return {
        "taxonomy": {c: list(f) for c, f in taxonomy.TAXONOMY.items()},
        "coarse_labels": list(taxonomy.COARSE_LABELS),
        "fine_labels": list(taxonomy.FINE_LABELS),
        "fine_idx_to_coarse_idx": list(taxonomy.FINE_IDX_TO_COARSE_IDX),
        "legacy_label_supervision": {k: list(v) for k, v in taxonomy.LEGACY_LABEL_SUPERVISION.items()},
        "staging_dir_supervision": {k: list(v) for k, v in taxonomy.STAGING_DIR_SUPERVISION.items()},
        "guidance_groups": [sorted(g) for g in taxonomy.GUIDANCE_GROUPS],
    }


def test_taxonomy_matches_snapshot() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert current_snapshot() == expected
