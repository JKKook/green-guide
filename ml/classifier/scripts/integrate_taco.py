"""TACO (Trash Annotations in Context) → 13 클래스 매핑 + bbox 크롭 ingest.

배경:
- TACO 1,500장 / 4,784 annotation (60 카테고리, 28 supercategory)
- 폴리곤 segmentation + bbox 제공 (라벨 정합성 자체는 95%+ 신뢰)
- in-the-wild 폐기물 사진 — 사용자 분포에 AI Hub 보다 가까움

처리 흐름:
1. annotations.json 의 각 annotation 에서 (image, bbox, category) 추출
2. category → 우리 13 클래스 매핑 (mapping 결정 csv 로 기록)
3. flickr_640_url (640px 작은 버전) 으로 이미지 다운로드 (스트리밍, 디스크 절약)
4. bbox 크롭 + 256x256 리사이즈 + JPG q=90 저장
5. data/raw/garbage-classification/{our_class}/taco_{id:06d}.jpg

사용:
    .venv/bin/python scripts/integrate_taco.py [--cap-per-class 200] [--limit 100]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from _base import PREPROCESSOR_ROOT, PROJECT_ROOT, RAW_DIR
from PIL import Image

TACO_ANNOTATIONS = Path("/tmp/TACO/data/annotations.json")
STORE_SIZE = 256
JPEG_QUALITY = 90

# TACO supercategory / category → 우리 13 클래스 매핑.
# (None 은 매핑 안 함 / 제외)
TACO_MAPPING: dict[str, str | None] = {
    # paper 계열
    "Magazine paper": "paper",
    "Tissues": "paper",
    "Normal paper": "paper",
    "Paper bag": "paper",
    "Plastified paper bag": "paper",  # 코팅이지만 종이 베이스
    "Paper straw": "paper",
    "Wrapping paper": "paper",

    # cardboard 계열 (Carton supercategory)
    "Toilet tube": "cardboard",
    "Other carton": "cardboard",
    "Egg carton": "cardboard",
    "Drink carton": "cardboard",  # 종이팩
    "Corrugated carton": "cardboard",
    "Meal carton": "cardboard",
    "Pizza box": "cardboard",
    "Paper cup": "cardboard",  # 종이컵 — paper or cardboard? 두꺼운 종이 → cardboard

    # glass
    "Glass bottle": "glass",
    "Glass jar": "glass",
    "Glass cup": "glass",
    "Broken glass": "glass",

    # metal
    "Aluminium foil": "metal",
    "Aerosol": "metal",
    "Food Can": "metal",
    "Drink can": "metal",
    "Metal bottle cap": "metal",
    "Metal lid": "metal",
    "Scrap metal": "metal",
    "Pop tab": "metal",

    # plastic (단단한 플라스틱)
    "Clear plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Disposable plastic cup": "plastic",
    "Other plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Disposable food container": "plastic",
    "Other plastic container": "plastic",
    "Squeezable tube": "plastic",
    "Other plastic": "plastic",

    # vinyl (얇은 비닐/필름)
    "Plastic film": "vinyl",
    "Six pack rings": "vinyl",
    "Garbage bag": "vinyl",
    "Other plastic wrapper": "vinyl",
    "Single-use carrier bag": "vinyl",
    "Polypropylene bag": "vinyl",
    "Crisp packet": "vinyl",
    "Plastic straw": "vinyl",

    # styrofoam
    "Foam cup": "styrofoam",
    "Foam food container": "styrofoam",
    "Styrofoam piece": "styrofoam",

    # food_waste
    "Food waste": "food_waste",

    # electronics (TACO 에선 배터리만)
    "Battery": "electronics",

    # etc (기타/잡종)
    "Cigarette": "etc",
    "Unlabeled litter": "etc",
    "Plastic glooves": "etc",  # 위생장갑 — 복합재
    "Plastic utensils": "etc",  # 일회용 식기
    "Rope & strings": "etc",
    "Aluminium blister pack": "etc",  # 약 포장재 — 복합재
    "Carded blister pack": "etc",
    "Shoe": "etc",  # 신발은 clothes 와 분리, etc 가 안전

    # 매핑 안 하는 카테고리: 없음 (모두 매핑 시도)
}


def download_image(url: str, max_retries: int = 2, timeout: int = 30) -> Image.Image | None:
    """flickr 이미지 스트리밍 다운로드. 실패 시 None."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content))
        except Exception:  # noqa: BLE001
            if attempt + 1 < max_retries:
                time.sleep(1)
    return None


def crop_bbox_and_save(
    img: Image.Image, bbox: tuple[float, float, float, float],
    out_path: Path, expand: float = 0.10,
) -> bool:
    """bbox 크롭 + 약간 확장 + 256 리사이즈 + JPG q=90 저장."""
    W, H = img.size
    x, y, w, h = bbox
    # 확장 (객체 컨텍스트 약간 포함)
    pad_w = w * expand
    pad_h = h * expand
    x0 = max(0, int(x - pad_w))
    y0 = max(0, int(y - pad_h))
    x1 = min(W, int(x + w + pad_w))
    y1 = min(H, int(y + h + pad_h))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return False  # 너무 작음
    crop = img.convert("RGB").crop((x0, y0, x1, y1))
    crop = crop.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=JPEG_QUALITY)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="TACO → 13 클래스 ingest")
    ap.add_argument("--cap-per-class", type=int, default=300,
                    help="클래스당 최대 ingest 장수 (기본 300)")
    ap.add_argument("--limit", type=int, default=0,
                    help="N>0 이면 처음 N annotation 만 처리 (테스트용)")
    ap.add_argument("--annotations", default=str(TACO_ANNOTATIONS),
                    help="TACO annotations.json 경로")
    ap.add_argument("--mapping-log",
                    default=str(PROJECT_ROOT / "diagnostics" / "taco_mapping_decisions.csv"),
                    help="매핑 결정 로그 CSV")
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        sys.exit(f"annotations.json 없음: {ann_path}\n"
                 f"먼저 git clone https://github.com/pedropro/TACO.git /tmp/TACO")

    print(f"[taco] {ann_path} 로드...")
    with ann_path.open() as fh:
        data = json.load(fh)
    images_by_id = {im["id"]: im for im in data["images"]}
    cats_by_id = {c["id"]: c["name"] for c in data["categories"]}
    print(f"  이미지 {len(data['images'])}장, 어노테이션 {len(data['annotations'])}개, "
          f"카테고리 {len(data['categories'])}개")

    # 매핑 안 된 카테고리 경고
    mapped_names = set(TACO_MAPPING.keys())
    unmapped = set(cats_by_id.values()) - mapped_names
    if unmapped:
        print(f"  ⚠ 매핑 정의 안 됨 ({len(unmapped)}개): {sorted(unmapped)}")
    print(f"  매핑 분포: {Counter(v for v in TACO_MAPPING.values() if v)}")

    # 클래스별 카운터 + 로그
    per_class_count = defaultdict(int)
    skip_count = Counter()
    mapping_log_rows = []

    # annotation 별로 처리 (이미지 다운로드는 캐싱)
    img_cache: dict[int, Image.Image | None] = {}
    anns = data["annotations"]
    if args.limit > 0:
        anns = anns[:args.limit]

    print(f"\n처리 시작 — {len(anns)} annotation, cap {args.cap_per_class}/class\n")
    t0 = time.time()
    saved = 0
    for i, ann in enumerate(anns, 1):
        cat_name = cats_by_id[ann["category_id"]]
        our_class = TACO_MAPPING.get(cat_name)
        if our_class is None:
            skip_count["unmapped"] += 1
            continue
        if per_class_count[our_class] >= args.cap_per_class:
            skip_count[f"cap_reached:{our_class}"] += 1
            continue

        img_id = ann["image_id"]
        if img_id not in img_cache:
            meta = images_by_id[img_id]
            url = meta.get("flickr_640_url") or meta.get("flickr_url") or meta.get("coco_url")
            img_cache[img_id] = download_image(url)
        img = img_cache[img_id]
        if img is None:
            skip_count["download_fail"] += 1
            continue

        # 640px 다운로드한 경우 bbox 도 비율로 변환
        meta = images_by_id[img_id]
        bbox = ann["bbox"]  # COCO format: [x, y, w, h]
        if meta.get("flickr_640_url") and img.size != (meta["width"], meta["height"]):
            sx = img.size[0] / meta["width"]
            sy = img.size[1] / meta["height"]
            bbox = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]

        out_path = RAW_DIR / our_class / f"taco_{ann['id']:06d}.jpg"
        if crop_bbox_and_save(img, bbox, out_path):
            per_class_count[our_class] += 1
            saved += 1
            mapping_log_rows.append({
                "annotation_id": ann["id"],
                "image_id": img_id,
                "taco_category": cat_name,
                "our_class": our_class,
                "bbox": bbox,
                "saved_path": str(out_path.relative_to(PREPROCESSOR_ROOT)),
            })
        else:
            skip_count["crop_too_small"] += 1

        if i % 100 == 0 or i == len(anns):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(anns)}] saved={saved} elapsed={elapsed:.0f}s "
                  f"cache={len(img_cache)}")

    print("\n=== 완료 ===")
    print(f"총 처리: {len(anns)} annotation, 저장 {saved}장")
    print("\n클래스별 ingest 결과:")
    for c in sorted(per_class_count.keys()):
        print(f"  {c:<14} {per_class_count[c]:>4}장")
    print("\nSkip 이유:")
    for reason, n in sorted(skip_count.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {n}")

    # 매핑 로그 저장
    log_path = Path(args.mapping_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "annotation_id", "image_id", "taco_category", "our_class", "bbox", "saved_path",
        ])
        w.writeheader()
        w.writerows(mapping_log_rows)
    print(f"\n매핑 로그: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
