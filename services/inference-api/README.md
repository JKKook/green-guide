---
title: GreenGuide Waste API
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# waste-api

GreenGuide AI 의 세 번째 서브 프로젝트. [`waste-classifier`](../waste-classifier) 가 학습·export 한 ONNX 모델을 FastAPI 로 감싸 HTTP 엔드포인트로 노출하는 추론 서버.

Flutter 앱·웹·다른 서비스 등 어디서든 사진 한 장만 보내면 6-class 폐기물 분류 결과를 받을 수 있다.

```
[클라이언트 (Flutter·웹·curl)]
        │
        │  POST /predict (이미지)
        ▼
[waste-api (FastAPI + ONNX Runtime)]
        │
        │  CNN classifier.onnx (43 MB)
        ▼
[6-class 분류 결과 JSON]
```

---

## 목차

1. [프로젝트 위치](#프로젝트-위치)
2. [핵심 결정 사항](#핵심-결정-사항)
3. [엔드포인트](#엔드포인트)
4. [설치 및 실행](#설치-및-실행)
5. [사용 예제](#사용-예제)
6. [실측 결과](#실측-결과)
7. [구성 요소](#구성-요소)
8. [테스트](#테스트)
9. [트러블슈팅](#트러블슈팅)
10. [알려진 한계와 향후 개선](#알려진-한계와-향후-개선)
11. [프로젝트 구조](#프로젝트-구조)

---

## 프로젝트 위치

```
GreenGuide AI
├── waste-preprocessor     (1) 수집·전처리·벡터화          완성
├── waste-classifier       (2) 지도학습 분류기 + ONNX      완성 (CNN 92.35%)
├── waste-api              (3) HTTP 추론 서버              현재
└── Flutter 클라이언트     (4) 모바일 앱                   예정
```

---

## 핵심 결정 사항

| 분야 | 선택 | 이유 |
|---|---|---|
| 프레임워크 | **FastAPI** | 비동기 지원, Pydantic 통합, OpenAPI 문서 자동 생성, 타입 안전 |
| 추론 엔진 | **ONNX Runtime** | Framework-agnostic, CPU/GPU 모두 지원, 가벼움 |
| 모델 | CNN (기본) — `WASTE_API_MODEL_PATH` 로 MLP 도 가능 | 정확도·크기 균형 |
| 모델 로드 | 서버 시작 시 1회 (lifespan event) | cold start 제거, 매 요청 빠름 |
| 이미지 처리 | Pillow | 다양한 포맷 자동 지원 (JPG·PNG·WebP 등) |
| CORS | `*` (개발 단계) | Flutter web·다른 origin에서 호출 가능. 배포 시 좁힐 것 |
| 인증 | 없음 (V1) | 학습 목적. production 에는 API Key·OAuth 추가 |

---

## 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 서비스 정보 (버전·모델 경로·클래스 목록) |
| `GET` | `/health` | 헬스체크 (모니터링용) |
| `GET` | `/labels` | 6개 클래스 목록 |
| `POST` | `/predict` | **이미지 업로드 → 분류 결과 (메인)**. 자동으로 Supabase `user_uploads` 에 기록되고 응답에 `upload_id` 반환 |
| `POST` | `/feedback` | 사용자 피드백 기록 (`upload_id` + confirmed/corrected + 올바른 라벨). 재학습 데이터로 활용 |
| `GET` | `/docs` | Swagger UI (자동 생성된 대화형 문서) |
| `GET` | `/redoc` | ReDoc (대안 문서 UI) |

### POST /feedback 스키마
```json
// 요청 (예측이 정확함)
{ "upload_id": "abc123", "confirmed": true }

// 요청 (예측이 틀렸음)
{ "upload_id": "abc123", "confirmed": false, "corrected_label": "plastic" }

// 응답
{ "upload_id": "abc123", "feedback_status": "confirmed", "feedback_label": "cardboard" }
```

### Active Learning Loop
사용자 사진 수집은 기본 활성. 환경변수로 끄기:
```bash
WASTE_API_COLLECT_UPLOADS=false python main.py
```
수집된 데이터는 [`../waste-classifier/retrain.py`](../waste-classifier/retrain.py) 로 모델 재학습에 사용.

### POST /predict 응답 스키마
```json
{
  "predicted_class": "cardboard",
  "predicted_index": 0,
  "confidence": 0.9996,
  "all_probabilities": {
    "cardboard": 0.9996,
    "glass": 0.0004,
    "metal": 1.15e-5,
    "paper": 1.55e-5,
    "plastic": 4.28e-6,
    "trash": 6.55e-6
  },
  "model_arch": "cnn",
  "inference_ms": 56.99
}
```

### 에러 응답
- `400 Bad Request` — 빈 파일 또는 디코딩 불가
- `413 Request Entity Too Large` — 10MB 초과
- `415 Unsupported Media Type` — JPG/PNG/WebP/BMP 외 형식

---

## 설치 및 실행

### 1. 가상환경 + 의존성
```bash
cd /Users/whdrnr01/ai/waste-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 전제 — 모델 파일
```bash
ls ../waste-classifier/outputs/models/cnn/classifier.onnx
```
없으면 waste-classifier 의 `python main.py all --arch cnn` 먼저 실행.

### 3. 서버 실행
```bash
# 개발 모드 (코드 변경 시 자동 재시작)
python main.py --reload

# 운영 모드 (다른 포트)
python main.py --host 0.0.0.0 --port 8000 --workers 4
```

기본 주소: http://localhost:8000

### 4. 환경 변수
| 변수 | 기본값 | 설명 |
|---|---|---|
| `WASTE_API_MODEL_PATH` | `../waste-classifier/outputs/models/cnn/classifier.onnx` | ONNX 모델 경로 (절대/상대 경로) |

예: MLP 모델로 전환
```bash
WASTE_API_MODEL_PATH=../waste-classifier/outputs/models/mlp/classifier.onnx \
  python main.py
```

---

## 사용 예제

### curl
```bash
# 서비스 정보
curl http://localhost:8000/

# 헬스체크
curl http://localhost:8000/health

# 라벨 목록
curl http://localhost:8000/labels

# 이미지 분류 (메인)
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/photo.jpg;type=image/jpeg"
```

### Python (httpx 또는 requests)
```python
import httpx

with open("photo.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/predict",
        files={"image": ("photo.jpg", f, "image/jpeg")},
    )

result = response.json()
print(f"{result['predicted_class']} ({result['confidence']:.1%})")
```

### Flutter (Dart) 미리보기
```dart
import 'package:http/http.dart' as http;

final request = http.MultipartRequest(
  'POST',
  Uri.parse('http://localhost:8000/predict'),
);
request.files.add(
  await http.MultipartFile.fromPath('image', '/path/to/photo.jpg'),
);
final response = await request.send();
final body = await response.stream.bytesToString();
print(body);  // JSON 응답
```

### Swagger UI 로 대화형 테스트
브라우저에서 http://localhost:8000/docs 접속 → "POST /predict" → "Try it out" → 이미지 업로드.

---

## 실측 결과

2026-05-17 기준.

### 응답 시간 (M1/M2 Mac, CPU, CNN 모델)
| 단계 | 시간 |
|---|---:|
| 이미지 디코딩 + 전처리 | ~5-10 ms |
| ONNX Runtime 추론 | ~40-60 ms |
| **전체 응답** | **~50-70 ms** |

### 분류 정확도 샘플
실제 Kaggle 데이터로 검증:

| 입력 이미지 | 예측 | 신뢰도 | 정답 |
|---|---|---:|---|
| `cardboard/cardboard1.jpg` | cardboard | **99.96%** | ✅ |
| `plastic/plastic1.jpg` | glass | 82.43% | ❌ (plastic 16.74%, 2위) |

CNN 의 전체 test accuracy 92.35% 와 일치하는 결과. plastic/glass 혼동은 학습 단계 confusion matrix 에서도 확인된 패턴.

---

## 구성 요소

| 파일 | 역할 | 핵심 |
|---|---|---|
| `src/core/` | 공통단 | `config`(모든 env 의 SSOT·`.env` 로드), `log`(로거), `singleton`(`@lazy_singleton`), `errors`(도메인 예외→HTTP) |
| `src/services/` | 비즈니스 로직 | `image_io`(읽기·검증·크롭), `cascade`(손/이진 게이트→분류), `regions_service`(다중재질 영역), `recording`(업로드 기록) |
| `src/routers/` | 얇은 라우터 | `meta`·`inference`·`admin`·`learning` — 입력 읽기 → 서비스 호출 → 응답 조립 |
| `src/api.py` | app 조립 | lifespan(모델 lazy load·정리 루프) + 미들웨어 + 라우터 include |
| `src/preprocess.py` | bytes → 텐서 | waste-preprocessor 와 **완전히 동일한** RGB→resize→normalize→reshape |
| `src/inference.py` 외 모델 모듈 | ONNX wrapper | `hier_inference`·`segment`·`clip_identity`·`dinov2_classifier`·`stage1_classifier`·`hand_detector` 등 |
| `src/schemas.py` | Pydantic 모델 | 자동 검증·OpenAPI 스키마 생성 |
| `main.py` | uvicorn 진입점 | host/port/reload/workers CLI 인자 |

---

## 테스트

```bash
.venv/bin/python -m pytest
```

총 **18개 테스트** (전처리 11, API 통합 7):

| 카테고리 | 검증 |
|---|---|
| `test_preprocess.py` | RGB 변환, grayscale 자동 변환, 잘못된 bytes 거부, 정규화 shape/dtype/범위, MLP/CNN reshape, end-to-end |
| `test_api.py` | 4개 엔드포인트 정상 응답, 실제 cardboard 샘플 정확 분류, 잘못된 Content-Type/empty/invalid bytes 거부 |

테스트는 실제 ONNX 모델을 로드하므로 waste-classifier 의 학습·export 완료가 전제.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `FileNotFoundError: ONNX 모델 파일을 찾을 수 없음` | classifier 학습 미완료 | `cd ../waste-classifier && .venv/bin/python main.py all --arch cnn` |
| `415 Unsupported Media Type` | Content-Type 헤더 누락/잘못됨 | curl `;type=image/jpeg` 명시 또는 클라이언트 헤더 확인 |
| `413 Request Entity Too Large` | 10MB 초과 | 클라이언트에서 리사이즈 후 전송, 또는 `config.MAX_UPLOAD_SIZE_BYTES` 조정 |
| 첫 요청만 느림 | 서버 시작 직후 모델 워밍업 | 정상 — lifespan 으로 로드되지만 첫 추론 시 ONNX session 캐시 발생 |
| Flutter web 에서 CORS 에러 | 다른 origin 호출 | 현재 `*` 허용. production 에선 `CORS_ORIGINS` 좁히기 |
| 분류 결과가 부정확 | CNN 92.35% 한계 | 데이터 증강·앙상블·더 큰 backbone (waste-classifier 개선) |

---

## 알려진 한계와 향후 개선

| 항목 | 현재 | 개선 방향 |
|---|---|---|
| 인증 | 없음 | API Key 헤더, JWT, OAuth |
| Rate limiting | 없음 | slowapi, Redis 기반 |
| 로깅 | stdout 만 | structlog, 요청별 trace ID |
| Supabase 연동 | 없음 | 예측 로그를 `predictions` 테이블에 INSERT |
| 배치 추론 | 단일 이미지 only | `/predict_batch` 추가 |
| 모델 hot reload | 재시작 필요 | 파일 변경 감지 + lifespan refresh |
| Docker | 없음 | Dockerfile + multi-stage build |
| 모니터링 | `/health` 만 | Prometheus metrics, OpenTelemetry |
| HTTPS | 없음 (uvicorn HTTP) | nginx/Caddy 리버스 프록시 |

---

## 프로젝트 구조

```
waste-api/
├── .gitignore
├── README.md
├── pytest.ini
├── requirements.txt
├── main.py                       # uvicorn CLI
├── ruff.toml                     # 린트 게이트 (ruff check src/ tests/)
├── docs/REFACTORING_GUIDE.md     # 구조 원칙·리팩토링 이력
├── src/
│   ├── api.py                    # app 조립 (lifespan·미들웨어·라우터 include)
│   ├── core/                     # 공통단: config · log · singleton · errors
│   ├── routers/                  # meta · inference · admin · learning
│   ├── services/                 # image_io · cascade · regions_service · recording
│   ├── preprocess.py             # bytes → tensor (preprocessor 와 동일 변환)
│   ├── inference.py              # ONNX Runtime 싱글톤 (+ hier_inference, segment, clip_identity …)
│   └── schemas.py                # Pydantic 요청/응답
└── tests/
    ├── conftest.py               # TestClient + 샘플 이미지 fixture
    ├── test_characterization.py  # 엔드포인트 경로·응답 필드 동결 (리팩토링 안전망)
    ├── test_preprocess.py
    └── test_api.py
```
