# waste_app — Flutter 모바일 앱 (그린가이드 AI)

> 소스: `waste_app/lib/`, `waste_app/pubspec.yaml`, `docs/design/GREENGUIDE_UIUX_SPEC.md` (2026-08-13 탐색)

**Android 전용** Flutter 앱 (`com.greenguide.waste_app`, iOS 미사용). git 아님. README는 V1(6클래스) 시점으로 낡음 — UI/UX의 as-built 정본은 `docs/design/GREENGUIDE_UIUX_SPEC.md`(2026-07-28 코드 실측).

## 구조

- **상태 관리: setState만** (MVP 규모 — Provider/BLoC 미도입). 전역 테마만 `ValueNotifier`. 영속: SharedPreferences(설정) + sqflite(히스토리).
- **화면**: 스플래시 → 온보딩(동의 게이트, 미동의 시 진입 불가) → 하단 4탭 셸 (**홈·스캔·기록·설정**). 결과는 별도 화면이 아닌 풀스크린 바텀시트 **ResultModal**(2,696줄, 최대 파일 — 증거 칩·다중재질 빗금 오버레이·탭-투-셀렉트·지역 카드·CAM 다이얼로그 전부 여기).
- **스마트 캡처**: `live_camera_screen` — 가속도계 안정도 100% 시 자동 트리거 + 품질 게이트(밝기·Laplacian 선명도). framing box는 시각 안내 전용(실제 분류는 풀프레임).
- 레거시: `result_screen.dart`는 완성됐으나 push 경로 없음(미정리).

## 추론 라우팅 (`services/prediction_service.dart`)

1. cloud 모드: `/predict-hier` 우선, 404/503이면 `/predict-centered`→`/predict` 격하
2. on-device 모드: `local_inference.dart`(onnxruntime 플러그인, Dart 전처리, **계층 게이트 온디바이스 적용**) → confidence <0.60이면 cloud 재판정 → "cloud 검증" 배지
3. 어려운 케이스가 자동으로 Supabase에 쌓여 active learning 우선 라벨링 대상이 되는 구조

## 모델 OTA (`services/remote_model_service.dart`)

**번들 ONNX 에셋 없음** — `/model/latest` → 버전 비교 → HF Hub에서 ONNX + taxonomy.json 다운로드·SHA256 검증. 앱 용량 204→88MB. 다운로드 전/실패 시 cloud로 동작.

## 클래스 메타 3단 폴백 (`services/class_loader.dart`)

서버 `/labels` → SharedPreferences 디스크 캐시 → 하드코딩 fallback(계층 대분류 14종; 구 6클래스 fallback은 2026-07 폐기). 클래스가 늘어나도 코드 수정 없이 반영.

## 신뢰도 판정 (`data/confidence.dart`)

top-1만이 아니라 **margin(top1−top2) + normalized entropy**로 high/medium/low 판정. reject 임계 0.55 + entropy>0.7.

## 테마 / 디자인

Material 3, 시드 `#2E7D32`(Green 800), 주아체(Jua) 디스플레이 폰트. `app_theme.dart` 실측값이 [waste-api](waste-api.md) `design/tokens.json`의 원천(역류 구조). UX 패턴: 햅틱 위계, 되돌리기 10단계, graceful degradation(재질·객체·지역·CAM은 선택적 향상), Trust UI(증거 칩·"왜 이렇게 분류했어?" CAM 설명).

## 특기

- `korea_map.dart`: 외부 SDK 없는 CustomPainter 시도 폴리곤 + point-in-polygon (원천 southkorea-maps, 자동 생성 — 수동 수정 금지)
- 개인정보: GPS EXIF 자동 제거 고지 + 동의 시트 1회
- 테스트: `widget_test.dart` 1개뿐 (실질 테스트 없음)

관련: [architecture-overview](architecture-overview.md) · [model-versions-accuracy](model-versions-accuracy.md)
