"""AI Hub 140 배경 추출 → non_object 학습 데이터. (Tier 2-1 보조)

생활폐기물 이미지에서 '객체 bbox 를 회피한 영역'을 크롭 = 배경/바닥/클러터 패치.
non_object 클래스의 '물체 없는 장면' 커버용 (손 데이터는 사용자 폰 사진이 정본).

사용:
    .venv/bin/python scripts/extract_bg_140.py --label-filesn 47732 \\
        --source-filesn 47740 47735 47742 47652 47741 47739 47682 --cap 700
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrate_aihub_140 import RAW_DIR, STAGING, _apikey, fetch_to_zip  # noqa: E402

STORE = 256
IMG_EXT = (".jpg", ".jpeg", ".png")
MAX_OVERLAP = 0.12   # 크롭이 객체 bbox 와 이만큼 미만 겹칠 때만 '배경'으로 채택


def parse_all_boxes(zip_path: Path) -> dict[str, list[tuple]]:
    """라벨 zip → {basename: [(x1,y1,x2,y2), ...]} — CLASS 무관 모든 BOX."""
    result: dict[str, list[tuple]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".json"):
                continue
            try:
                d = json.loads(z.read(name))
            except Exception:  # noqa: BLE001
                continue
            fn = d.get("FILE NAME") or d.get("FILE_NAME")
            if not fn:
                continue
            boxes = []
            for bb in d.get("Bounding", []):
                try:
                    x1, y1, x2, y2 = (int(float(bb[k])) for k in ("x1", "y1", "x2", "y2"))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
                except (KeyError, ValueError, TypeError):
                    continue
            result[os.path.basename(fn)] = boxes
    return result


def _overlap_frac(crop: tuple, box: tuple) -> float:
    cx0, cy0, cx1, cy1 = crop
    bx0, by0, bx1, by1 = box
    ix = max(0, min(cx1, bx1) - max(cx0, bx0))
    iy = max(0, min(cy1, by1) - max(cy0, by0))
    inter = ix * iy
    area = (cx1 - cx0) * (cy1 - cy0)
    return inter / area if area else 1.0


def bg_crops(img: Image.Image, boxes: list[tuple], n: int, rng: random.Random) -> list[Image.Image]:
    W, H = img.size
    out: list[Image.Image] = []
    for _ in range(n * 8):  # 시도 횟수
        if len(out) >= n:
            break
        s = rng.randint(int(min(W, H) * 0.3), int(min(W, H) * 0.55))
        if s < 64:
            continue
        x = rng.randint(0, W - s); y = rng.randint(0, H - s)
        crop = (x, y, x + s, y + s)
        if all(_overlap_frac(crop, b) < MAX_OVERLAP for b in boxes):
            out.append(img.crop(crop).resize((STORE, STORE), Image.BILINEAR))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Hub 140 배경 → non_object")
    ap.add_argument("--label-filesn", type=int, required=True)
    ap.add_argument("--source-filesn", type=int, nargs="+", required=True)
    ap.add_argument("--cap", type=int, default=700)
    ap.add_argument("--crops-per-img", type=int, default=2)
    args = ap.parse_args()

    apikey = _apikey()
    STAGING.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    out_dir = RAW_DIR / "non_object"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = sum(1 for p in out_dir.glob("aihubbg_*"))
    print(f"배경 추출 시작 (기존 aihubbg {start}장, cap {args.cap})")

    print(f"\n[1] 라벨(fileSn={args.label_filesn}) 다운로드 + 전체 bbox 파싱...")
    lw = STAGING / "bg_label"
    labels = parse_all_boxes(fetch_to_zip(args.label_filesn, lw, apikey))
    print(f"  라벨 이미지 {len(labels)}개")
    shutil.rmtree(lw, ignore_errors=True)

    n = start
    for i, fsn in enumerate(args.source_filesn, 1):
        if n >= args.cap:
            break
        print(f"\n[2.{i}] 원천 fileSn={fsn} (누적 {n}/{args.cap})...")
        sw = STAGING / f"bg_{fsn}"
        try:
            zp = fetch_to_zip(fsn, sw, apikey)
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if n >= args.cap or not name.lower().endswith(IMG_EXT):
                        if n >= args.cap:
                            break
                        continue
                    boxes = labels.get(os.path.basename(name), [])
                    try:
                        im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                    except Exception:  # noqa: BLE001
                        continue
                    for c in bg_crops(im, boxes, args.crops_per_img, rng):
                        if n >= args.cap:
                            break
                        c.save(out_dir / f"aihubbg_non_object_{n:06d}.jpg", quality=88)
                        n += 1
            print(f"  → 누적 {n}장")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] fileSn={fsn} 실패: {exc}")
        finally:
            shutil.rmtree(sw, ignore_errors=True)

    print(f"\n✓ 배경 {n - start}장 추가 → non_object 총 "
          f"{sum(1 for p in out_dir.iterdir() if p.suffix.lower() in IMG_EXT)}장")
    shutil.rmtree(STAGING, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
