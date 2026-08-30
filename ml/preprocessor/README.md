# greenguide-preprocessor

GreenGuide AI의 첫 서브 프로젝트. Kaggle의 폐기물 분류 데이터셋을 자동으로 수집·정제·전처리·벡터화하고, 결과를 로컬 디스크와 Supabase(Postgres + Storage)에 저장하는 이미지 전처리 파이프라인이다.

이후 단계인 지도학습 분류기(차기 서브 프로젝트)가 바로 입력으로 사용할 수 있는 형태의 학습 데이터를 만든다.

---

## 목차

1. [프로젝트 위치](#프로젝트-위치)
2. [핵심 결정 사항](#핵심-결정-사항)
3. [아키텍처](#아키텍처)
4. [파이프라인 단계](#파이프라인-단계)
5. [설치 및 환경 설정](#설치-및-환경-설정)
6. [사용법](#사용법)
7. [데이터 저장 위치](#데이터-저장-위치)
8. [실행 결과](#실행-결과)
9. [결과 검증 방법](#결과-검증-방법)
10. [테스트](#테스트)
11. [알려진 한계와 향후 개선](#알려진-한계와-향후-개선)
12. [트러블슈팅](#트러블슈팅)
13. [프로젝트 구조](#프로젝트-구조)

---

## 프로젝트 위치

상위 비전인 **GreenGuide AI** (분리수거·폐기물 사진 자동 분류 AI 에이전트)의 데이터 파이프라인 중 가장 첫 단계.

```
GreenGuide AI (전체 비전)
├── greenguide-preprocessor      <-- 현재 프로젝트 (전처리)
├── 데이터 로더             (다음)
├── 분류 모델 학습          (다음)
├── ReAct 에이전트화        (장기)
└── 배포·모니터링           (장기)
```

이 프로젝트는 **모델 학습을 직접 수행하지는 않는다**. 학습기에 넣기 좋은 형태의 정규화된 벡터·메타데이터를 만들어 두는 역할이다.

---

## 핵심 결정 사항

| 분야 | 선택 | 이유 |
|---|---|---|
| 데이터셋 | Kaggle "Garbage Classification" (6 classes, 2,527장) | 입문에 적절한 크기, 라벨 명확, 공개 |
| 실행 환경 | venv + .py 모듈 | 실무 표준에 가까움, 배포·패키징 용이 |
| 이미지 크기 | 224 × 224 × 3 | ImageNet 사전학습 모델과 호환 |
| 정규화 | ImageNet mean/std | 사전학습 모델 전이학습 시 표준 관행 |
| 벡터화 방식 | Flatten 1D (150,528 dim) | 사용자 의도에 따른 Traditional ML / FC NN 지원 |
| 저장 정밀도 | float16 + gzip 압축 (`.npz`) | 1.5GB → 200MB (81% 절감), 정확도 손실 거의 없음 |
| 메타·이미지 저장 | Supabase (Postgres + Storage) | Free tier 충분, 추후 클라우드 확장 베이스 |
| 벡터 본체 저장 | 로컬 파일시스템 | Free tier 용량 초과, 학습 시 mmap 친화 |

---

## 아키텍처

```
+--------------------------------------------------------------+
|  로컬 머신                                                   |
|                                                              |
|  data/raw/garbage-classification/   <- Kaggle 원본           |
|  data/processed/vectors/*.npz       <- float16 압축 벡터     |
|  data/processed/manifest.json       <- 전체 메타·통계        |
|                                                              |
|  src/  Python 모듈 파이프라인                                |
|     collect -> catalog -> cleanse -> preprocess -> vectorize |
|                                                              |
+---------------------------+----------------------------------+
                            |
                            | supabase-py SDK
                            v
+--------------------------------------------------------------+
|  Supabase (클라우드, Free Tier)                              |
|                                                              |
|  Storage Bucket "raw-images" (Public)                        |
|    cardboard/<id>.jpg, glass/<id>.jpg, ...                   |
|                                                              |
|  Postgres "public.items"                                     |
|    id | label | original_url | vector_path | stats | ...     |
|                                                              |
+--------------------------------------------------------------+
```

---

## 파이프라인 단계

전체 흐름은 `greenguide_preprocessor/pipeline.py:run()` 에서 오케스트레이션된다.

| # | 단계 | 모듈 | 입력 | 출력 |
|---|---|---|---|---|
| 1 | 수집 | `collect.py` | Kaggle slug | `data/raw/garbage-classification/<label>/*.jpg` |
| 2 | 카탈로그 | `catalog.py` | raw 디렉토리 | `data/interim/catalog.json` |
| 3 | 클렌징 | `cleanse.py` | catalog | `data/interim/catalog.cleansed.json` |
| 4 | 전처리 | `preprocess.py` | 개별 이미지 | numpy 배열 (224,224,3) float32 |
| 5 | 벡터화 | `vectorize.py` | 전처리 결과 | `.npz` (float16 압축) |
| 6 | 업로드 (선택) | `storage.py` | 이미지 + 메타 | Supabase Storage + Postgres |
| 7 | manifest | `pipeline.py` | 위 결과 통합 | `data/processed/manifest.json` |

### 1. 수집 (`collect.py`)
- Kaggle CLI 자동 다운로드 (kaggle.json 인증 필요)
- 이미 받아져 있으면 다운로드 생략
- 둘 다 안 되면 수동 다운로드 안내 후 종료
- Kaggle zip 의 중첩 폴더 구조를 자동 평탄화

### 2. 카탈로그 (`catalog.py`)
- 6개 클래스 폴더를 스캔해 모든 이미지 파일을 발견
- 각 이미지에 12자 UUID 부여 후 라벨·경로와 함께 JSON 으로 저장

### 3. 클렌징 (`cleanse.py`)
- **손상 이미지 필터**: PIL 의 `Image.verify()` 로 검사
- **중복 이미지 제거**: perceptual hash (`imagehash.phash`, hash_size=8) 비교
- 통계 출력: input/corrupt/duplicate/kept

### 4. 전처리 (`preprocess.py`)
순서가 중요하다.
1. `load_rgb`: 이미지 로드 후 RGB 가 아니면 변환 (L, RGBA 등 → RGB)
2. `resize_square`: 224×224 로 bilinear 리사이즈
3. `to_normalized_array`: uint8 [0,255] → float32 [0,1] → ImageNet 정규화

### 5. 벡터화 (`vectorize.py`)
- `flatten`: (224, 224, 3) → (150528,) 1D
- `save_vector`: float32 → float16 다운캐스트 후 `np.savez_compressed`
- `load_vector`: 자동으로 float32 로 복원

### 6. 업로드 (`storage.py`)
- 이미지: `raw-images/<label>/<id>.<ext>` 로 업로드, public URL 획득
- 메타: `items` 테이블에 upsert (chunk size 100)

### 7. manifest 작성
- 처리 결과를 통합해 `data/processed/manifest.json` 생성
- 실패한 항목은 `failed` 배열에 사유와 함께 기록

---

## 설치 및 환경 설정

### Python 환경
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # ../greenguide-common 이 editable 로 함께 설치됨
```

> 이미지 전처리·Supabase 접속·분류체계는 [`greenguide-common`](../greenguide-common/README.md) 공통단을 사용한다.
> Apple Silicon 에서 Rosetta 셸(x86_64)로 실행하면 arm64 휠과 충돌하므로 `arch -arm64 .venv/bin/python ...` 으로 실행.

### Kaggle 인증 (자동 다운로드용)
1. `pip install kaggle` (이미 설치되어 있을 수 있음)
2. https://www.kaggle.com/settings/account 접속
3. **API Tokens** 섹션에서 `Create New Token`
4. 받은 `kaggle.json` 을 다음 형식으로 작성/저장:
   ```bash
   mkdir -p ~/.kaggle
   echo '{"username":"YOUR_USERNAME","key":"YOUR_TOKEN"}' > ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   - 신형 KGAT 토큰(`KGAT_...`)도 그대로 `key` 필드에 넣으면 작동
   - username 은 본인 Kaggle 프로필 URL 의 마지막 부분

### Supabase 설정
1. https://supabase.com 에서 프로젝트 생성 (또는 기존 프로젝트 사용)
2. **Project Settings → API**
   - Project URL: `https://<project-ref>.supabase.co`
   - `service_role` key (JWT, `eyJ...` 형식) ← anon key 아님 주의
3. **Storage** → New bucket
   - Name: `raw-images`
   - Public bucket: **체크 필수**
4. **SQL Editor** 에서 다음 실행:
   ```sql
   create table public.items (
       id              text primary key,
       label           text not null,
       original_url    text not null,
       vector_path     text not null,
       source_path     text not null,
       filename        text not null,
       stats           jsonb,
       created_at      timestamptz default now()
   );

   alter table public.items enable row level security;
   ```
5. `.env` 파일 작성:
   ```bash
   cp .env.example .env
   ```
   ```ini
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_KEY=<service_role JWT>
   SUPABASE_BUCKET=raw-images
   SUPABASE_TABLE=items
   ```

### 보안 주의
- `.env` 는 git 에 커밋하지 말 것 (`.gitignore` 에 이미 포함)
- `service_role` key 는 RLS 를 우회하므로 절대 frontend·공개 저장소에 노출 금지
- 노출된 키는 Supabase Dashboard 에서 즉시 폐기/재발급

---

## 사용법

### CLI

```bash
# 환경 정보·디렉토리 확인
python main.py

# 개별 단계만 실행
python main.py --step collect          # 데이터셋 다운로드
python main.py --step catalog          # 카탈로그 생성
python main.py --step cleanse          # 손상·중복 제거
python main.py --step supabase-check   # Supabase 연결 확인

# 전체 파이프라인 (로컬만)
python main.py

# 전체 파이프라인 + Supabase 업로드
python main.py --upload
```

### Python API
```python
from greenguide_preprocessor.pipeline import run

# 전체 실행
manifest_path = run(upload_to_supabase=True)

# 처리된 벡터 1개 로드 (자동으로 float32 로 복원)
from greenguide_preprocessor.vectorize import load_vector
vec = load_vector("b2dfb128a3ad")    # shape: (150528,), dtype: float32

# manifest 직접 읽기
import json
with open("data/processed/manifest.json") as f:
    manifest = json.load(f)
for item in manifest["items"][:3]:
    print(item["label"], item["id"], item["original_url"])

# Supabase 에서 메타 조회
from greenguide_preprocessor.storage import _client
c = _client()
rows = c.table("items").select("*").eq("label", "plastic").limit(10).execute().data
```

---

## 데이터 저장 위치

### 로컬

| 경로 | 용도 | 예상 크기 |
|---|---|---|
| `data/raw/garbage-classification/<label>/` | Kaggle 원본 이미지 | ~46 MB |
| `data/interim/catalog.json` | 1차 카탈로그 (수집 직후) | ~440 KB |
| `data/interim/catalog.cleansed.json` | 클렌징 후 카탈로그 | ~440 KB |
| `data/processed/vectors/<id>.npz` | float16 압축 1D 벡터 | ~80 KB × 2,522 = 200 MB |
| `data/processed/manifest.json` | 전체 통합 메타 | ~1.3 MB |

### Supabase

| 위치 | 내용 | 접근 |
|---|---|---|
| Storage `raw-images/<label>/<id>.<ext>` | 원본 이미지 (public) | URL 직접 / supabase-py |
| Postgres `public.items` | 메타데이터 + 통계 | SQL / supabase-py |

`items` 테이블 컬럼:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | text PK | 12자 UUID |
| `label` | text | 6개 클래스 중 하나 |
| `original_url` | text | Supabase Storage public URL |
| `vector_path` | text | 로컬 `.npz` 상대 경로 |
| `source_path` | text | 원본 이미지 상대 경로 |
| `filename` | text | 원본 파일명 |
| `stats` | jsonb | `{mean, std, min, max}` 픽셀 통계 |
| `created_at` | timestamptz | 입력 시각 (default now()) |

---

## 실행 결과

2026-05-17 실측 기준.

### 처리 수치
```
input    : 2,527       (Kaggle 원본 총합)
cleansed : 2,522       (중복 5장 제거, 손상 0장)
processed: 2,522
failed   : 0
```

### 클래스별 분포
```
cardboard:  403
glass    :  501
metal    :  409   (-1, 중복)
paper    :  592   (-2)
plastic  :  480   (-2)
trash    :  137
총합     : 2,522
```

### 압축 효과 (벡터)
| 방식 | 파일당 | 총 크기 |
|---|---:|---:|
| 만약 .npy float32 | 588 KB | ~1.5 GB |
| 실제 .npz float16 (적용) | ~80 KB | **200 MB** |
| 절감률 | | **86%** |

float16 다운캐스트로 인한 정확도 손실: rtol < 1e-2 (사실상 무시 가능).

### 인프라
- Supabase Postgres `items`: 2,522 rows
- Supabase Storage `raw-images`: 2,522 files (~80 MB)
- 로컬 벡터: 2,522 `.npz` (200 MB)

---

## 결과 검증 방법

### 1. manifest 통계로 픽셀 분포 확인
```python
import json, numpy as np
m = json.load(open("data/processed/manifest.json"))
stats = np.array([[i["stats"]["mean"], i["stats"]["std"]] for i in m["items"]])
print(f"전체 픽셀 mean 평균 : {stats[:,0].mean():+.4f}")
print(f"전체 픽셀 std  평균 : {stats[:,1].mean():+.4f}")
```

ImageNet mean/std 는 ImageNet 데이터셋 기준이므로, 이 데이터에 적용한 결과의 평균은 0/1 에서 약간 시프트된다 (실측: mean≈+0.84, std≈+0.84). 사전학습 모델과의 호환을 위한 표준 관행이며 학습에는 문제 없음.

### 2. 단일 샘플 추적 (원본 → 벡터 → 복원)
```python
import numpy as np
from PIL import Image
from greenguide_preprocessor.vectorize import load_vector
from greenguide_preprocessor import config

vec = load_vector("b2dfb128a3ad")
arr = vec.reshape(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)

# 역정규화 후 이미지로 복원
mean = np.array(config.IMAGENET_MEAN)
std  = np.array(config.IMAGENET_STD)
denorm = np.clip((arr * std + mean) * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(denorm).save("recon.jpg")
```

### 3. Supabase 직접 확인
- Dashboard → Table Editor → `items` 에서 row 시각 확인
- Dashboard → Storage → `raw-images` 에서 라벨별 이미지 갤러리
- public URL 을 브라우저에 직접 붙여넣어 이미지 열람 가능

### 4. 클래스 균형 점검
```python
from collections import Counter
import json
m = json.load(open("data/processed/manifest.json"))
print(Counter(i["label"] for i in m["items"]))
```

---

## 테스트

```bash
.venv/bin/python -m pytest
```

총 16개 테스트 (catalog 5, preprocess 6, vectorize 5).

| 영역 | 검증 항목 |
|---|---|
| `catalog` | 카탈로그 생성, 라벨별 항목 수, ID 유일성, 디렉토리 누락 처리, JSON roundtrip |
| `preprocess` | 비-RGB 변환, 리사이즈 크기, 정규화 값 범위, 출력 shape/dtype, 파일 누락 처리, 통계 키 |
| `vectorize` | 차원, 형상 검증, 자동 float32 캐스팅, .npz roundtrip, float16 저장 확인 |

---

## 알려진 한계와 향후 개선

| 항목 | 현재 한계 | 개선 방향 |
|---|---|---|
| ID 생성 방식 | 매 실행마다 새 UUID → 재실행 시 고아 파일 누적 | `hash(label + filename)` 기반 deterministic ID 도입 |
| Resumability | 항상 처음부터 재처리 | `.npz` 존재 시 스킵 로직 추가 |
| 정규화 통계 | ImageNet 통계 차용 (이 데이터셋과 미세 미스매치) | 데이터셋 자체 mean/std 계산해 캐시 |
| 검증 도구 | 매번 ad-hoc 스크립트 | `python main.py --inspect <id>` CLI 화 |
| 시각화 | 텍스트 통계만 | matplotlib 히스토그램·클래스별 샘플 격자 노트북 |
| Kaggle 잔여 파일 | zip 안의 부가 파일이 raw 폴더에 남음 | collect 단계에서 자동 정리 |
| 코드 품질 | 자동 린트 없음 | ruff, mypy 도입 |

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `[collect] kaggle CLI not found` | venv 비활성 상태 | `source .venv/bin/activate` 후 재실행 |
| `Could not find the table 'public.items'` | 테이블 미생성 또는 schema cache 지연 | SQL 재실행 또는 `notify pgrst, 'reload schema';` |
| `Bucket not found` | bucket 미생성 또는 이름 불일치 | Dashboard → Storage 에서 `raw-images` 생성 확인 |
| `SUPABASE_URL not set` | `.env` 없음 또는 키 누락 | `.env` 위치(`프로젝트 루트`), 키 이름 확인 |
| Supabase 에 권한 오류 | `anon` key 사용 중 | `service_role` key (JWT, `eyJ...`) 로 교체 |
| 재실행 시 `.npz` 파일이 두 배 | 매 실행 새 UUID 발급 (위 한계 참조) | manifest 와 매칭되지 않는 `.npz` 수동 삭제 |

---

## 프로젝트 구조

```
greenguide-preprocessor/
├── .env                          # 실제 자격증명 (gitignore)
├── .env.example                  # 자격증명 양식
├── .gitignore
├── README.md
├── pytest.ini
├── requirements.txt
├── main.py                       # CLI 진입점
├── data/
│   ├── raw/                      # Kaggle 원본 (gitignore)
│   ├── interim/                  # 카탈로그·중간 산출물 (gitignore)
│   └── processed/                # 최종 산출물 (gitignore)
│       ├── vectors/              #   <id>.npz
│       └── manifest.json
├── src/
│   ├── __init__.py
│   ├── config.py                 # 전역 상수·환경변수 로드
│   ├── collect.py                # Phase 1: 데이터 수집
│   ├── catalog.py                # Phase 2: 카탈로그 JSON
│   ├── cleanse.py                # Phase 3: 손상·중복 제거
│   ├── preprocess.py             # Phase 4: 리사이즈/RGB/정규화
│   ├── vectorize.py              # Phase 5: Flatten + 압축 저장
│   ├── storage.py                # Phase 6: Supabase 클라이언트
│   └── pipeline.py               # Phase 7: 오케스트레이션
└── tests/
    ├── __init__.py
    ├── conftest.py               # 공통 fixture
    ├── test_catalog.py
    ├── test_preprocess.py
    └── test_vectorize.py
```

---

## 참고

- 데이터셋: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Supabase: https://supabase.com
- ImageNet 정규화 통계: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
