"""/predict-with-regions 영역 응답에 빗금과 같은 color_hex 가 실린다 (feature/api-region-colors)."""
from __future__ import annotations

from src import schemas


def test_material_region_carries_color_hex() -> None:
    r = schemas.MaterialRegion(slug="metal", bbox_norm=[0, 0, 1, 1], avg_conf=0.9,
                               cell_count=3, color_hex="#8E9AAF")
    assert r.model_dump()["color_hex"] == "#8E9AAF"


def test_material_region_color_optional_for_old_clients() -> None:
    r = schemas.MaterialRegion(slug="metal", bbox_norm=[0, 0, 1, 1], avg_conf=0.9, cell_count=3)
    assert r.color_hex is None
