"""AI Hub 데이터셋 통합 — --dataset 으로 71362(기본) / 140 선택.

[71362] '재활용품 분류 및 선별 데이터' — 클래스별 청크 처리.
검출형(bbox) 데이터를 객체 크롭 → 256px → raw/{our_class}/ 로 저장.
디스크 절약을 위해:
  - zip 을 메모리에서 entry 단위로 읽음 (전체 추출 안 함)
  - 단일 part 는 rename, 다중 part 만 concat
  - 처리 후 모든 임시 파일 즉시 삭제
  - per-class cap 도달 시 조기 종료

사전: AIHUB_APIKEY 환경변수 (waste-preprocessor/.env).

사용:
    .venv/bin/python scripts/integrate_aihub.py \
        --our-class vinyl --source-filesn 482446 --label-filesn 482488 --cap 10000

[140] '생활 폐기물 이미지'(dataSetSn 140) — 다중 품목 → 단일 우리 클래스.
140 은 71362 와 라벨 형식이 다름:
  - 라벨 JSON: {"FILE NAME": "x.jpg", "Bounding": [{"CLASS": "전자제품",
    "Drawing": "BOX", "x1","y1","x2","y2"}, ...]}  ← 코너좌표(x1y1x2y2)
  - 라벨링데이터.zip 1개가 전 카테고리 공통 (Training 47869 / Validation 47732)
  - 원천데이터는 품목별 zip (전자제품/컴퓨터.zip 등)
  - 이미지는 실제 배출현장 사진(클러터 많음) → **반드시 bbox 크롭** 필요

전략: 라벨zip 다운로드 → CLASS==<aihub-class> 박스만 {basename: [boxes]} 로 수집 →
품목별 원천 zip 을 스트리밍하며 해당 박스 크롭 → 256px → raw/{our_class}/, 전역 cap.

사용:
    .venv/bin/python scripts/integrate_aihub.py --dataset 140 \
        --our-class electronics --aihub-class 전자제품 \
        --label-filesn 47869 --cap 1000 --per-source-cap 200 \
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

from _base import PROJECT_ROOT, RAW_DIR
from PIL import Image

from waste_common import settings  # noqa: E402

STAGING = settings.WASTE_ROOT / "ml" / "data" / "raw" / "aihub"
STAGING_140 = settings.WASTE_ROOT / "ml" / "data" / "raw" / "aihub_140"
STORE_SIZE = 256   # 저장 해상도 (학습은 224 로 resize)
IMG_EXT = (".jpg", ".jpeg", ".png")


def _apikey() -> str:
    import waste_common.settings  # noqa: F401,PLC0415 — .env 로드
    key = os.getenv("AIHUB_APIKEY")
    if not key:
        sys.exit("ERROR: AIHUB_APIKEY 미설정 (waste-preprocessor/.env)")
    return key


def fetch_to_zip(filesn: int, work_dir: Path, apikey: str, dataset_key: int = 71362) -> Path:
    """curl | tar 스트리밍 추출 (download.tar 저장 안 함 → peak 디스크 반감) →
    내부 .part 를 (단일)rename / (다중)concat-삭제 → .zip 경로 반환."""
    work_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://api.aihub.or.kr/down/0.6/{dataset_key}.do?fileSn={filesn}"
    print(f"  ⬇ streaming fileSn={filesn} (curl | tar)...")

    # curl stdout → tar stdin 파이프 (전체 tar 를 디스크에 안 올림)
    curl = subprocess.Popen(
        ["curl", "-s", "-L", "-H", f"apikey:{apikey}", url],
        stdout=subprocess.PIPE,
    )
    tar = subprocess.Popen(
        ["tar", "-x", "-C", str(work_dir)],
        stdin=curl.stdout,
    )
    if curl.stdout:
        curl.stdout.close()  # tar 가 EOF/SIGPIPE 받을 수 있게
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
        prefix = str(p).rsplit(".part", 1)[0]
        groups[prefix].append(p)

    prefix, plist = next(iter(groups.items()))
    zip_path = Path(prefix)
    plist.sort(key=lambda p: int(str(p).rsplit(".part", 1)[1]))

    if len(plist) == 1:
        plist[0].rename(zip_path)  # rename — 추가 공간 0
    else:
        # concat — 각 part 를 이어붙인 즉시 삭제 (peak ≈ 전체 + 1 part)
        with zip_path.open("wb") as out:
            for p in plist:
                with p.open("rb") as f:
                    shutil.copyfileobj(f, out)
                p.unlink()
    return zip_path


def parse_labels(zip_path: Path) -> dict[str, list[list[float]]]:
    """라벨 zip → {image_filename: [[x,y,w,h], ...]}."""
    result: dict[str, list[list[float]]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".json"):
                continue
            try:
                data = json.loads(z.read(name))
            except Exception:  # noqa: BLE001
                continue
            fn = data.get("IMAGE_INFO", {}).get("FILE_NAME")
            if not fn:
                continue
            boxes = []
            for ann in data.get("ANNOTATION_INFO", []):
                if ann.get("SHAPE_TYPE") == "BOX" and ann.get("POINTS"):
                    boxes.append(ann["POINTS"][0])  # [x, y, w, h]
            result[fn] = boxes
    return result


def process_source(
    zip_path: Path,
    labels: dict[str, list[list[float]]],
    our_class: str,
    cap: int,
    start_count: int,
) -> int:
    """source zip 의 이미지를 메모리에서 읽어 bbox 크롭 → 256px → 저장.
    Returns: 누적 저장 수."""
    out_dir = RAW_DIR / our_class
    out_dir.mkdir(parents=True, exist_ok=True)
    n = start_count

    with zipfile.ZipFile(zip_path) as z:
        names = [x for x in z.namelist() if x.lower().endswith(IMG_EXT)]
        for name in names:
            if n >= cap:
                break
            base = os.path.basename(name)
            try:
                im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            boxes = labels.get(base, [])
            if boxes:
                for box in boxes:
                    if n >= cap:
                        break
                    try:
                        x, y, w, h = box[:4]
                        crop = im.crop((int(x), int(y), int(x + w), int(y + h)))
                        if crop.width < 16 or crop.height < 16:
                            continue
                        crop = crop.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
                        crop.save(out_dir / f"aihub_{our_class}_{n:06d}.jpg", quality=88)
                        n += 1
                    except Exception:  # noqa: BLE001
                        continue
            else:
                # 라벨 없으면 전체 이미지 (fallback)
                im2 = im.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
                im2.save(out_dir / f"aihub_{our_class}_{n:06d}.jpg", quality=88)
                n += 1
    return n


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


def process_source_140(zip_path: Path, labels: dict[str, list[tuple]], our_class: str,
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
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Hub 71362/140 → 클래스별 통합")
    ap.add_argument("--dataset", type=int, choices=(71362, 140), default=71362,
                    help="AI Hub dataSetSn (기본 71362)")
    ap.add_argument("--our-class", required=True, help="우리 클래스 slug (예: vinyl)")
    ap.add_argument("--source-filesn", type=int, nargs="+", required=True,
                    help="원천데이터 fileSn (71362: 1개, 140: 품목별 여러 개)")
    ap.add_argument("--label-filesn", type=int, required=True, help="라벨링데이터 fileSn")
    ap.add_argument("--cap", type=int, default=None,
                    help="클래스당 최대 저장 수 (기본 71362: 10000, 140: 1000)")
    ap.add_argument("--keep-existing", action="store_true",
                    help="[71362] 기존 aihub_ 파일 유지 (기본: 누적)")
    ap.add_argument("--aihub-class", help="[140] 라벨 CLASS 값 (예: 전자제품)")
    ap.add_argument("--per-source-cap", type=int, default=250, help="[140] 원천 zip 당 상한")
    args = ap.parse_args()

    if args.dataset == 140:
        if not args.aihub_class:
            ap.error("--dataset 140 은 --aihub-class 필요")
        if args.cap is None:
            args.cap = 1000
        return main_140(args)
    if len(args.source_filesn) != 1:
        ap.error("--dataset 71362 는 --source-filesn 1개")
    args.source_filesn = args.source_filesn[0]
    if args.cap is None:
        args.cap = 10000
    return main_71362(args)


def main_71362(args: argparse.Namespace) -> int:
    apikey = _apikey()
    STAGING.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"AI Hub 통합 — {args.our_class} (cap={args.cap})")
    print("=" * 60)

    existing = count_existing(args.our_class)
    aihub_existing = sum(
        1 for p in (RAW_DIR / args.our_class).glob("aihub_*")
    ) if (RAW_DIR / args.our_class).exists() else 0
    print(f"  현재 {args.our_class}: 총 {existing}장 (그 중 aihub {aihub_existing}장)")
    if existing >= args.cap:
        print(f"  이미 cap({args.cap}) 도달 — skip")
        return 0

    # 1) 라벨 다운로드 + 파싱 (작음)
    print("\n[1/3] 라벨 다운로드 + 파싱...")
    label_work = STAGING / "label_work"
    label_zip = fetch_to_zip(args.label_filesn, label_work, apikey)
    labels = parse_labels(label_zip)
    print(f"  라벨 {len(labels)}개 이미지 분량 파싱")
    shutil.rmtree(label_work, ignore_errors=True)

    # 2) 원천 다운로드 + 병합 (스트리밍)
    print("\n[2/3] 원천 이미지 스트리밍 다운로드 (수 GB, 시간 소요)...")
    src_work = STAGING / "source_work"
    src_zip = fetch_to_zip(args.source_filesn, src_work, apikey)
    print(f"  zip 크기: {src_zip.stat().st_size / 1e9:.1f} GB")

    # 3) 크롭 + 저장 (메모리 스트리밍)
    print("\n[3/3] bbox 크롭 + 256px 저장...")
    final = process_source(src_zip, labels, args.our_class, args.cap, existing)
    added = final - existing
    print(f"  추가: {added}장 → {args.our_class} 총 {final}장")

    # 정리
    shutil.rmtree(src_work, ignore_errors=True)
    print("\n✓ 완료. staging 정리됨.")
    print("  다음: 다른 클래스도 통합 후 retrain.py 실행")
    return 0


def main_140(args: argparse.Namespace) -> int:
    apikey = _apikey()
    STAGING_140.mkdir(parents=True, exist_ok=True)
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
    lw = STAGING_140 / "label_work"
    label_zip = fetch_to_zip(args.label_filesn, lw, apikey, dataset_key=140)
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
        sw = STAGING_140 / f"src_{fsn}"
        try:
            src_zip = fetch_to_zip(fsn, sw, apikey, dataset_key=140)
            n = process_source_140(src_zip, labels, args.our_class, args.cap,
                               args.per_source_cap, n)
            print(f"  → 누적 {n}장")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] fileSn={fsn} 처리 실패: {exc}")
        finally:
            shutil.rmtree(sw, ignore_errors=True)

    print(f"\n✓ 완료: {args.our_class} {existing} → {n}장 (+{n - existing})")
    shutil.rmtree(STAGING_140, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
