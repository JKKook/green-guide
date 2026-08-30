# 데이터셋 스테이징 디렉터리

> 소스: `aihub_71385_staging/`, `aihub_71647_staging/`, `openimages_staging/`, `synth_indoor_staging/`, `taco_staging/`, `bin/aihubshell` (2026-08-13 탐색)

외부 데이터셋을 다운로드→크롭→[waste-preprocessor](waste-preprocessor.md) `fine-staging/`으로 통합하기 전의 작업 영역. 루트에 위치.

| 디렉터리 | 데이터셋 | 크기 | 상태 |
|---|---|---|---|
| `aihub_71385_staging` | AI-Hub 71385 생활폐기물 활용·환류 + AI-Hub 140 | ~14G | **크롭 12.7만 + fine-staging 통합 완료** |
| `taco_staging` | TACO (실외 쓰레기, CC BY) | 2.3G | 크롭 ~2,053 완료, `taco_*` prefix 통합됨 |
| `openimages_staging` | Open Images V6/V7 bbox 채굴 | 6.8G | 1·2차 크롭 완료, `oi_`/`oi2_` prefix 통합됨 |
| `synth_indoor_staging` | 자체 실내 배경 합성 | 329M | 19라벨×300=5,700장 산출, **병합 보류(의도적 격리)** |
| `aihub_71647_staging` | AI-Hub 029 물체조작 손동작 3D | 139M | raw, 미착수(손 mask 후보, 신청 승인 대기 이력) |
| `aihub_staging` | — | 0 | 빈 폴더(폐기된 초기 시도) |

## aihub_71385_staging (핵심)

- `crop_71385.py`: fileSn tar 다운로드 → zip 파트 병합 → **truncated zip 순회로 중단돼도 받은 만큼 salvage** → 라벨 bbox 크롭(`crops/<class>/<cond>/`, cond=clean/multi/dirty/dirty_multi). 44개 zip 매핑(재활용선별장 A / 실내형분류기 B / 어플리케이션 C).
- `crop_140.py`: AI-Hub 140 품목 zip → light_bulb(전구≠형광등 오분리 방지), glass_deposit(소주·맥주병), electronics, cardboard 수혈.
- `collector.py` + `overnight_c.sh`: 클래스별 쿼터(`TARGETS`) 달성까지 야간 반복 수집, 신규 <300 3회면 은퇴.
- `integrate_staging.py`: crops → fine-staging 통합. PER_CLASS_MAX 10,000, **도메인 우선순위 C(스마트폰) > B(실내) > A(선별장)**, 조건은 파일명 보존 `aihub385_<cond>__*.jpg`.
- API 키는 `waste-preprocessor/.env`의 `AIHUB_APIKEY`. `bin/aihubshell`은 AI-Hub 공식 CLI(v0.6)이지만 실제 스크립트는 salvage 목적으로 동일 엔드포인트를 직접 curl.

## openimages_staging

- 1차 `collect_openimages.py`: Tin can→metal_boost, Plastic bag→taco_vinyl, Wine glass→glass_clear.
- 2차 `collect_openimages2.py`: 의류·book·box·주방금속·전자기기 증량 + **범용 Bottle 37,411 bbox를 CLIP 제로샷으로 재질 라우팅**(waste-api의 clip_identity 재사용, 불확실은 폐기 — "독립 신호라 자기강화 아님").

## synth_indoor_staging

생성기는 `waste-classifier/scripts/synthesize_indoor_scenes.py` — 실사용 사진의 저saliency 영역에서 배경 수확 + fine-staging 크롭을 u2netp 누끼로 합성. `synmo_` prefix는 frozen test 자동 제외. 규율: **학습 사이클 도중 fine-staging 수정 금지**. 참고: v9의 합성 실내 접근은 배경 편향으로 폐기됨 → [data-experiments](data-experiments.md).

관련 메모리: AI-Hub 71385 수집은 Supabase 쿼터 원칙과 함께 진행 중이었음 → [supabase-infra](supabase-infra.md).
