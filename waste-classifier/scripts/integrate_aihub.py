"""AI Hub '재활용품 분류 및 선별 데이터'(71362) 통합 — 클래스별 청크 처리.

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
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"
STAGING = PROJECT_ROOT.parent / "aihub_staging"
DATASET_KEY = 71362
STORE_SIZE = 256   # 저장 해상도 (학습은 224 로 resize)
IMG_EXT = (".jpg", ".jpeg", ".png")


def _apikey() -> str:
    import os as _os
    from dotenv import load_dotenv
    load_dotenv(PREPROCESSOR_ROOT / ".env")
    key = _os.getenv("AIHUB_APIKEY")
    if not key:
        sys.exit("ERROR: AIHUB_APIKEY 미설정 (waste-preprocessor/.env)")
    return key


def fetch_to_zip(filesn: int, work_dir: Path, apikey: str) -> Path:
    """curl | tar 스트리밍 추출 (download.tar 저장 안 함 → peak 디스크 반감) →
    내부 .part 를 (단일)rename / (다중)concat-삭제 → .zip 경로 반환."""
    work_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do?fileSn={filesn}"
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
        sys.exit(f"ERROR: {work_dir} 에서 part/zip 못 찾음")

    from collections import defaultdict
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


def count_existing(our_class: str) -> int:
    d = RAW_DIR / our_class
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Hub 71362 → 클래스별 통합")
    ap.add_argument("--our-class", required=True, help="우리 클래스 slug (예: vinyl)")
    ap.add_argument("--source-filesn", type=int, required=True, help="원천데이터 fileSn")
    ap.add_argument("--label-filesn", type=int, required=True, help="라벨링데이터 fileSn")
    ap.add_argument("--cap", type=int, default=10000, help="클래스당 최대 저장 수")
    ap.add_argument("--keep-existing", action="store_true",
                    help="기존 aihub_ 파일 유지 (기본: 누적)")
    args = ap.parse_args()

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
    print(f"\n✓ 완료. staging 정리됨.")
    print(f"  다음: 다른 클래스도 통합 후 retrain.py 실행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
