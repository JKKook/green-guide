"""AI Hub '생활 폐기물 이미지'(dataSetSn 140) 통합 — 다중 품목 → 단일 우리 클래스.

140 은 71362 와 라벨 형식이 다름:
  - 라벨 JSON: {"FILE NAME": "x.jpg", "Bounding": [{"CLASS": "전자제품",
    "Drawing": "BOX", "x1","y1","x2","y2"}, ...]}  ← 코너좌표(x1y1x2y2)
  - 라벨링데이터.zip 1개가 전 카테고리 공통 (Training 47869 / Validation 47732)
  - 원천데이터는 품목별 zip (전자제품/컴퓨터.zip 등)
  - 이미지는 실제 배출현장 사진(클러터 많음) → **반드시 bbox 크롭** 필요

전략: 라벨zip 다운로드 → CLASS==<aihub-class> 박스만 {basename: [boxes]} 로 수집 →
품목별 원천 zip 을 스트리밍하며 해당 박스 크롭 → 256px → raw/{our_class}/, 전역 cap.

사용:
    .venv/bin/python scripts/integrate_aihub_140.py \\
        --our-class electronics --aihub-class 전자제품 \\
        --label-filesn 47869 --cap 1000 --per-source-cap 200 \\
        --source-filesn 47777 47773 47775 47781 47763 47783 47762 47766 47774 47785 47764
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"
STAGING = PROJECT_ROOT.parent / "aihub_staging_140"
DATASET_KEY = 140
STORE_SIZE = 256
IMG_EXT = (".jpg", ".jpeg", ".png")


def _apikey() -> str:
    from dotenv import load_dotenv
    load_dotenv(PREPROCESSOR_ROOT / ".env")
    key = os.getenv("AIHUB_APIKEY")
    if not key:
        sys.exit("ERROR: AIHUB_APIKEY 미설정")
    return key


def fetch_to_zip(filesn: int, work_dir: Path, apikey: str) -> Path:
    """curl | tar 스트리밍 → 내부 .part (단일)rename / (다중)concat → .zip 경로."""
    work_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do?fileSn={filesn}"
    print(f"  ⬇ streaming fileSn={filesn} (curl | tar)...")
    curl = subprocess.Popen(["curl", "-s", "-L", "-H", f"apikey:{apikey}", url],
                            stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "-x", "-C", str(work_dir)], stdin=curl.stdout)
    if curl.stdout:
        curl.stdout.close()
    tar_rc = tar.wait()
    curl_rc = curl.wait()
    if tar_rc != 0 or curl_rc != 0:
        raise RuntimeError(f"download/extract 실패 (curl={curl_rc}, tar={tar_rc})")

    parts = sorted(work_dir.rglob("*.part*"))
    if not parts:
        zips = list(work_dir.rglob("*.zip"))
        if zips:
            return zips[0]
        raise RuntimeError(f"{work_dir} 에서 part/zip 못 찾음")
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in parts:
        groups[str(p).rsplit(".part", 1)[0]].append(p)
    prefix, plist = next(iter(groups.items()))
    zip_path = Path(prefix)
    plist.sort(key=lambda p: int(str(p).rsplit(".part", 1)[1]))
    if len(plist) == 1:
        plist[0].rename(zip_path)
    else:
        with zip_path.open("wb") as out:
            for p in plist:
                with p.open("rb") as f:
                    shutil.copyfileobj(f, out)
                p.unlink()
    return zip_path


def parse_labels_140(zip_path: Path, aihub_class: str) -> dict[str, list[tuple]]:
    """라벨 zip → {basename: [(x1,y1,x2,y2), ...]} — CLASS==aihub_class 인 BOX 만."""
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
                if str(bb.get("CLASS", "")).strip() != aihub_class:
                    continue
                try:
                    x1, y1, x2, y2 = (int(float(bb[k])) for k in ("x1", "y1", "x2", "y2"))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
                except (KeyError, ValueError, TypeError):
                    continue
            if boxes:
                result[os.path.basename(fn)] = boxes
    return result


def process_source(zip_path: Path, labels: dict[str, list[tuple]], our_class: str,
                   cap: int, per_cap: int, start: int) -> int:
    out_dir = RAW_DIR / our_class
    out_dir.mkdir(parents=True, exist_ok=True)
    n = start
    taken_here = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(IMG_EXT):
                continue
            if n >= cap or taken_here >= per_cap:
                break
            boxes = labels.get(os.path.basename(name))
            if not boxes:
                continue
            try:
                im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            for (x1, y1, x2, y2) in boxes:
                if n >= cap or taken_here >= per_cap:
                    break
                try:
                    crop = im.crop((x1, y1, x2, y2))
                    if crop.width < 24 or crop.height < 24:
                        continue
                    crop = crop.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
                    crop.save(out_dir / f"aihub_{our_class}_{n:06d}.jpg", quality=88)
                    n += 1
                    taken_here += 1
                except Exception:  # noqa: BLE001
                    continue
    return n


def count_existing(our_class: str) -> int:
    d = RAW_DIR / our_class
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Hub 140 → 단일 클래스 통합")
    ap.add_argument("--our-class", required=True)
    ap.add_argument("--aihub-class", required=True, help="라벨 CLASS 값 (예: 전자제품)")
    ap.add_argument("--label-filesn", type=int, required=True)
    ap.add_argument("--source-filesn", type=int, nargs="+", required=True)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--per-source-cap", type=int, default=250)
    args = ap.parse_args()

    apikey = _apikey()
    STAGING.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"AI Hub 140 통합 — {args.our_class} ← CLASS={args.aihub_class} (cap={args.cap})")
    print("=" * 60)

    existing = count_existing(args.our_class)
    print(f"  현재 {args.our_class}: {existing}장")
    if existing >= args.cap:
        print("  이미 cap 도달 — skip")
        return 0

    # 1) 라벨 다운로드 + 파싱 (전자제품 박스만)
    print(f"\n[1] 라벨(fileSn={args.label_filesn}) 다운로드 + 파싱 (CLASS={args.aihub_class})...")
    lw = STAGING / "label_work"
    label_zip = fetch_to_zip(args.label_filesn, lw, apikey)
    labels = parse_labels_140(label_zip, args.aihub_class)
    print(f"  {args.aihub_class} 박스 보유 이미지: {len(labels)}개")
    shutil.rmtree(lw, ignore_errors=True)
    if not labels:
        sys.exit("ERROR: 해당 CLASS 라벨 0개 — aihub-class 값 확인")

    # 2) 품목별 원천 스트리밍 → 크롭 (전역 cap 까지)
    n = existing
    for i, fsn in enumerate(args.source_filesn, 1):
        if n >= args.cap:
            break
        print(f"\n[2.{i}] 원천 fileSn={fsn} 처리 (누적 {n}/{args.cap})...")
        sw = STAGING / f"src_{fsn}"
        try:
            src_zip = fetch_to_zip(fsn, sw, apikey)
            n = process_source(src_zip, labels, args.our_class, args.cap,
                               args.per_source_cap, n)
            print(f"  → 누적 {n}장")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] fileSn={fsn} 처리 실패: {exc}")
        finally:
            shutil.rmtree(sw, ignore_errors=True)

    print(f"\n✓ 완료: {args.our_class} {existing} → {n}장 (+{n - existing})")
    shutil.rmtree(STAGING, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
