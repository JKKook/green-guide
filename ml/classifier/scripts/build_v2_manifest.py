#!/usr/bin/env python3
"""분류 체계 v2(TAXONOMY_V2.md §3·§4-3) 학습 매니페스트 — 원본에 이미 있는 구분만 되살린다.

입력: 현행 학습 아이템(hier_dataset.build_hier_items — legacy manifest + fine-staging), 그대로 읽기만 한다.
근거(파일명·원본 라벨):
- aihub385_<cond>__ (crop_71385.py CODE_MAP): clean→clean, dirty→dirty, dirty_multi→dirty,
  multi(다중포장재)→상태 미상(분리 가능 포장 — composite 도 clean 도 아님, rule 에 표시). metal_boost 의 aihub385 는 c_3 캔류 → metal_can.
- taco_<annotation id> (collect_taco.py): annotations.json 원 카테고리로 metal_can/metal_scrap, Broken glass→broken.
  학습셋에 없는(소형 스킵) Broken glass 는 --recrop 으로 staging 밖에 다시 자른다.
- oi_tin_can / oi2_fork·spoon·knife (collect_openimages*.py), oi4hn_* 는 crops4_hardneg/manifest.csv 의 oi_label.
그 밖의 상태는 비워 둔다(미상). 금속·음식물 세부를 원본으로 못 가르면 fine_v2 를 비우고 미확정으로 센다.

출력: <out>/manifest_v2.csv (path, coarse, fine_v2, state, source, rule) + summary_v2.json.
사용: GREENGUIDE_PREPROCESSOR_ROOT=<메인 트리>/ml/preprocessor python scripts/build_v2_manifest.py \
        --raw <메인 트리>/ml/data/raw --out outputs/v2_manifest [--recrop <스크래치 폴더>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
from greenguide_common.taxonomy import FINE_TO_COARSE

from greenguide_classifier import config
from greenguide_classifier.hier_dataset import build_hier_items

AIHUB_STATE = {"clean": ("clean", "aihub_cond_clean"), "dirty": ("dirty", "aihub_cond_dirty"),
               "dirty_multi": ("dirty", "aihub_cond_dirty_multi"), "multi": ("", "aihub_cond_multi_separable")}  # 리더 판정 11-03: 분리 가능 포장 → 상태 미상(상태 손실 제외)
TACO_CAN = {"Drink can", "Food Can", "Aerosol", "Pop tab"}
TACO_SCRAP = {"Metal bottle cap", "Metal lid", "Scrap metal", "Aluminium foil"}
OI_CAN = {"tin_can", "tin_can2", "beer", "beer2"}
OI_SCRAP = {"frying_pan", "wok", "pressure_cooker", "serving_tray", "fork", "spoon", "knife"}


def taco_categories(raw: Path) -> tuple[dict[int, str], dict]:
    d = json.loads((raw / "taco" / "annotations.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    return {a["id"]: cats[a["category_id"]] for a in d["annotations"]}, d


def oi4_labels(raw: Path) -> dict[str, str]:
    with (raw / "openimages" / "crops4_hardneg" / "manifest.csv").open(encoding="utf-8") as f:
        return {Path(r["file"]).name: r["oi_label"] for r in csv.DictReader(f)}


def classify(it: dict, taco: dict[int, str], oi4: dict[str, str]) -> dict:
    sp = it["source_path"]
    name = Path(sp).name
    coarse = it["sup_slug"] if it["sup_kind"] == "coarse" else FINE_TO_COARSE[it["sup_slug"]]
    fine = it["sup_slug"] if it["sup_kind"] == "fine" else ""
    state, rules, src = "", [], it["origin"]

    if m := re.match(r"aihub385_([a-z_]+?)__", name):
        src = "aihub385"
        state, r = AIHUB_STATE[m.group(1)]
        rules.append(r)
        if coarse == "metal":
            fine, _ = "metal_can", rules.append("aihub_c3_can")
    elif m := re.match(r"taco_(\d+)\.jpg$", name):
        src = "taco"
        cat = taco.get(int(m.group(1)), "?")
        rules.append(f"taco:{cat}")
        if coarse == "metal":
            fine = "metal_can" if cat in TACO_CAN else "metal_scrap" if cat in TACO_SCRAP else ""
        if cat == "Broken glass":
            state = "broken"
        elif cat == "Aluminium foil":
            state = "clean"  # 순수 알루미늄 호일 = metal_scrap/clean (리더 판정 11-03, 은박 코팅 종이와 구분)
    elif m := re.match(r"oi4hn_(.+?)_\d+\.jpg$", name):
        src = "oi4"
        lab = oi4.get(name, m.group(1))
        rules.append(f"oi4:{lab}")
        if coarse == "metal":
            fine = "metal_can" if lab in OI_CAN else "metal_scrap" if lab in OI_SCRAP else ""
    elif m := re.match(r"(oi\d?)_(.+?)_\d+\.jpg$", name):
        src, lab = m.group(1), m.group(2)
        rules.append(f"{src}:{lab}")
        if coarse == "metal":
            fine = "metal_can" if lab in OI_CAN else "metal_scrap" if lab in OI_SCRAP else ""
    elif name.startswith("synmo_"):
        src = "synmo"
    elif sp.startswith("data/raw/garbage-classification/"):
        src = "legacy_gc"

    if coarse == "metal" and fine == "metal":
        fine = ""  # v2 에서 metal 세부는 can/scrap 둘뿐 — 원본 근거가 없으면 미확정
    if coarse == "food_waste" and fine == "food_waste" and src != "aihub385":
        fine = ""  # food_waste vs food_inedible 원본 구분 없음 → 미확정
    return {"path": sp, "coarse": coarse, "fine_v2": fine, "state": state, "source": src,
            "rule": ";".join(rules)}


def recrop_broken(raw: Path, d: dict, have: set[int], out: Path, min_side: int, pad: float) -> list[dict]:
    from PIL import Image  # noqa: PLC0415

    cats = {c["id"]: c["name"] for c in d["categories"]}
    meta = {im["id"]: im for im in d["images"]}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in d["annotations"]:
        if cats[a["category_id"]] != "Broken glass" or a["id"] in have:
            continue
        im = meta[a["image_id"]]
        p = raw / "taco" / "images" / Path(im["file_name"]).name
        if not p.exists():
            p = raw / "taco" / "images" / f"{a['image_id']:05d}.jpg"
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        sx, sy = img.size[0] / im["width"], img.size[1] / im["height"]
        x, y, w, h = a["bbox"][0] * sx, a["bbox"][1] * sy, a["bbox"][2] * sx, a["bbox"][3] * sy
        if min(w, h) < min_side:
            continue
        box = (max(0, int(x - w * pad)), max(0, int(y - h * pad)),
               min(img.size[0], int(x + w * (1 + pad))), min(img.size[1], int(y + h * (1 + pad))))
        f = out / f"taco_{a['id']:06d}.jpg"
        img.crop(box).save(f, quality=92)
        rows.append({"path": str(f), "coarse": "glass", "fine_v2": "", "state": "broken", "source": "taco_recrop",
                     "rule": f"taco:Broken glass;recrop_min{min_side}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="ml/data/raw (taco·openimages 원본)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--recrop", type=Path, help="학습셋에 없는 TACO Broken glass 를 여기에 다시 자름(staging 밖)")
    ap.add_argument("--recrop-min-side", type=int, default=32)
    a = ap.parse_args()

    taco, taco_raw = taco_categories(a.raw)
    oi4 = oi4_labels(a.raw)
    items = build_hier_items()
    rows = [classify(it, taco, oi4) for it in items]
    if a.recrop:
        have = {int(m.group(1)) for r in rows if r["source"] == "taco" and (m := re.match(r".*taco_(\d+)\.jpg$", r["path"]))}
        rows += recrop_broken(a.raw, taco_raw, have, a.recrop, a.recrop_min_side, 0.12)

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "manifest_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "coarse", "fine_v2", "state", "source", "rule"])
        w.writeheader()
        w.writerows(rows)
    by_fine = Counter(f"{r['coarse']}/{r['fine_v2'] or '-'}" for r in rows)
    by_state = Counter(f"{r['coarse']}/{r['state'] or '미상'}" for r in rows)
    undecided = Counter(r["coarse"] for r in rows if r["coarse"] in ("metal", "food_waste") and not r["fine_v2"])
    summary = {"preprocessor_root": str(config.PREPROCESSOR_ROOT), "n": len(rows),
               "by_source": dict(Counter(r["source"] for r in rows)),
               "state_total": dict(Counter(r["state"] or "미상" for r in rows)),
               "metal_fine": {k: v for k, v in by_fine.items() if k.startswith("metal/")},
               "food_fine": {k: v for k, v in by_fine.items() if k.startswith("food_waste/")},
               "undecided_required_fine": dict(undecided),
               "multi_state_unknown": sum("aihub_cond_multi_separable" in r["rule"] for r in rows),
               "by_coarse_fine": dict(sorted(by_fine.items())), "by_coarse_state": dict(sorted(by_state.items()))}
    (a.out / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "state_total", "metal_fine", "food_fine",
                                               "undecided_required_fine", "multi_state_unknown")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
