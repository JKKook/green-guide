# 스마트 촬영 기반 정확도 향상 전략

> 작성: 2026-05-30 / 활성 모델 `v20260525_150909` (12클래스, ResNet18 → ONNX 3-output) 기준.
> 청사진([GREENGUIDE_BLUEPRINT.md](GREENGUIDE_BLUEPRINT.md)) Tier 1/3/4 와 직접 연결됨.
> UI/UX 고도화를 잠시 멈추고 실사용 정확도(현재 측정 63.4%) 를 끌어올릴 방안을 정리.

---

## 1. 현재 ONNX 학습·추론 흐름

### 학습 (waste-classifier)
- **백본**: `torchvision.models.resnet18(weights=IMAGENET1K_V1)` 을 분리수거 12클래스로 fine-tune
- **헤드**: `CamWasteClassifierCNN` 으로 감싸서 `forward()` 가 `(logits, cam, embedding)` 3-output 동시 반환 — CAM·OOD·재학습 시그널을 1회 forward 로 모두 확보하려는 설계
- **입력 파이프라인**: PIL → EXIF 회전 보정 → 224×224 → ImageNet mean/std 정규화
- **학습 데이터 소스**:
  - Kaggle Garbage Classification (베이스)
  - AI Hub 71362 재활용품 / 140 생활폐기물 (도메인 보강)
  - `user_uploads` (active learning 피드백, frozen test 분포는 학습에서 제외)
- **클래스 가중치**: inverse-freq 로 불균형 보정 (e.g. etc 가 다른 클래스의 1/100 수준)
- **진단 게이트**: frozen held-out test 의 정확도가 직전 active 대비 회귀하면 publish 자동 차단

### ONNX export
- `CamWasteClassifierCNN.forward` 통째로 export → output 3개 (`logits[1,12]`, `cam[1,12,7,7]`, `pooled[1,512]`)
- export 직후 진단 1회 더 (numpy vs torch 결과 일치 검증)
- 이 ONNX 1 파일을 **클라우드/온디바이스 양쪽에서 동일** 하게 사용

### 추론 경로
| 경로 | 동작 |
|---|---|
| **온디바이스** (Flutter) | 앱에 ONNX 번들(`assets/models/classifier.onnx`) → onnxruntime 으로 `logits` 만 사용. 신뢰도 < 0.60 이면 cloud fallback |
| **클라우드** (waste-api) | 같은 ONNX 로 `/predict`(logits), `/predict-with-cam`(CAM), `/predict-with-regions`(셀별 argmax + u2netp 마스크), `/segment`(u2netp 누끼) 4개 엔드포인트 제공 |

### 갱신 루프
사용자 피드백 → `user_uploads` 적재 → retrain → 게이트 통과 시 `model_versions.is_active=true` 갱신 → 클라이언트가 자동으로 새 ONNX 다운로드.

---

## 2. 왜 실사용은 여전히 멀게 느껴지는가 — 측정값 재확인

청사진 Tier 1-1 에 이미 기록된 사실:

| 분포 | 정확도 | 차이 |
|---|---|---|
| frozen held-out (AI Hub 분포) | **95.9%** | 기준 |
| 실사용 (`user_uploads` 피드백 41건) | **63.4%** | **−32.5pp** |

이 32pp 갭의 원인은 "모델이 본 적 없는 분포":
- 손에 든 물체 (학습엔 거의 없음)
- 실내 잡배경 (학습은 흰 배경/스튜디오)
- 폰 카메라 거리·각도의 다양성
- 작은 주변기기(마우스·이어폰) 등 소형 전자제품

즉 모델 측면에선 **데이터를 채우는 것 외엔 근본 해결 없음**. 스마트 촬영은 데이터를 직접 채우진 못하지만 **잡음을 줄이고 학습 루프 입력 품질을 높이는** 두 역할을 함.

---

## 3. 스마트 촬영을 활용한 정확도 향상 방안

영향도 순.

### A. 촬영 시점 품질 게이트 (블러/저조도 사전 차단)
- **현재**: `image_quality.dart` 가 캡처 *후* 라플라시안 분산/평균 밝기로 판단 → 결과 모달에 배너만 띄움
- **개선**: 라이브 프리뷰 프레임에서 실시간으로 같은 지표 계산 → 품질 미달이면 **stability trigger 발동을 잠그고** "더 밝은 곳으로", "초점이 안 맞아요" 가이드. 품질 OK 일 때만 3초 카운트 시작
- **비용**: 작음 (간단한 픽셀 통계, GPU 불필요)
- **기대**: 흔들림·저조도 캡처 자체가 사라짐 → 노이즈 floor 제거. 실측 안 했지만 **+3~5pp** 추정
- **위치**: [waste_app/lib/services/stability_detector.dart](waste_app/lib/services/stability_detector.dart), [waste_app/lib/screens/live_camera_screen.dart](waste_app/lib/screens/live_camera_screen.dart), [waste_app/lib/data/image_quality.dart](waste_app/lib/data/image_quality.dart)

### B. u2netp 객체 자동 분리 + 크롭 후 분류 (가장 큰 잠재력)
- **아이디어**: 캡처 후 u2netp 으로 객체 마스크 추출 → 마스크 bbox 로 타이트하게 크롭 → 224 리사이즈 → 분류기에 넣음
- **효과**: 잡배경·손 픽셀 자체가 분류기 입력에서 사라짐. 실사용 갭의 큰 부분(배경 노이즈) 직접 해소
- **걸림돌**: 분류기가 **크롭/마스킹된 이미지를 본 적 없음** → 그대로 적용하면 도메인 미스매치로 오히려 떨어질 수 있음
- **해결 방법**:
  - **(B-1)** 학습 데이터에 **u2netp 으로 자동 크롭한 버전 50% 혼합** → 분류기가 양쪽 모두 익숙해짐. 비용 중간, 효과 큼
  - **(B-2)** 원본 + 크롭 둘 다 추론 → softmax 평균 (앙상블). 추론 비용 2배지만 재학습 필요 없음 → 빠른 실험에 적합
- **기대**: 손·배경 케이스에 대해 **+5~10pp** 잠재
- **위치**: [waste-api/src/segment.py](waste-api/src/segment.py), [waste-classifier/train.py](waste-classifier/train.py) (B-1 의 경우 학습 변경)

### C. 다중 프레임 캡처 + 앙상블 (TTA at capture)
- **아이디어**: stability 도달 시 1프레임 대신 **1초간 3~5프레임** 자동 캡처 → 각 프레임 분류 → softmax 평균. 프레임 간 top1 이 다르면 reject ("자세 살짝 바꿔 다시 찍어주세요")
- **효과**: crop-instability 와 단일 프레임 운 의존을 직접 완화. 청사진 Tier 3-1 의 캡처 측 구현
- **비용**: 카메라 burst + 추론 N회 (온디바이스 ResNet18 N회는 ~수백ms 추가)
- **기대**: **+2~4pp**, 신뢰도 보정 효과는 더 큼 (uncertain 케이스가 명확해짐)

### D. 캡처 전 in-distribution 체크 (라이브 프리뷰 분류)
- **아이디어**: 프리뷰에서 **5~10 FPS 로 분류 실행** → 화면에 top1 라벨/신뢰도 미리 표시. 사용자가 "지금 모델은 X 라고 본다" 를 보면서 구도 조정 → 만족스러우면 캡처
- **효과**: 사용자가 *모델이 보기 쉬운 자세* 를 능동적으로 찾게 됨. 가장 강력한 데이터 보강 도구이기도 함 (사용자가 좋은 캡처만 보냄)
- **비용**: 온디바이스 연속 추론 — Android ResNet18 5 FPS 는 가능하지만 발열·배터리 영향. 스마트 캡처 모드에만 한정하면 OK
- **기대**: 정량 어렵지만 **사용자 의지력이 가장 잘 발휘되는 안전장치**

### E. 객체 존재·중앙 정렬 게이트
- **아이디어**: stability 외에 **u2netp 마스크가 화면 중앙에서 면적 ≥ X%** 일 때만 자동 트리거
- **효과**: "손만 든 빈 화면", "물체 가장자리에 끼임" 같은 잘못된 캡처를 원천 차단
- **비용**: u2netp 라이브 프레임 추론은 무거움 → 200~300ms 간격 다운샘플 권장
- **기대**: 캡처 실패율 감소 (직접적 정확도 향상보단 사용자 시간 절약)

### F. 캡처 가이드 강화 (UX-only)
- "물체로 박스를 채우세요" 시각 가이드, "배경이 복잡해요" 휴리스틱 경고 등
- 비용은 작지만 **사용자가 가이드를 따를 때만** 효과. 청사진 Tier 4-1 그대로

---

## 4. 권장 우선순위와 솔직한 한계

### 추천 순서
1. **A (품질 게이트)** — 1~2일 작업, 회귀 위험 없음, 즉시 노이즈 제거
2. **C (다중 프레임 앙상블)** — 1주 작업, crop-instability 해소
3. **B-2 (원본+크롭 추론 앙상블)** — 1주 작업, 도메인 갭 일부 즉시 완화 (분류기 재학습 없이)
4. **D (라이브 프리뷰 분류)** — 2주 작업, 데이터 루프 자체를 강화
5. **B-1 (학습에 크롭 혼합)** — retrain 1회, A/B/C 적용 후 측정해보고 결정

### 솔직한 한계
- A~F 합쳐서 기대 가능한 향상은 **+10~15pp 정도** (63 → 75% 부근)
- 그 이상은 **모델이 "손에 든 마우스" 같은 케이스를 학습한 적 없음** 이라는 근본 문제 → 청사진 Tier 1-2 (실사용 데이터 직접 수집·학습) 외엔 답 없음
- **smart capture 는 갭을 좁히지만 닫진 못함**. 데이터 작업과 병행이 정답이라는 결론은 변하지 않음

### 측정 인프라 한 가지 보강
어떤 개선이 효과 있었는지 알려면 [waste-classifier/realworld_eval.py](waste-classifier/realworld_eval.py) 가 매 retrain 직후 자동으로 돌도록 파이프라인에 묶고, frozen vs realworld 정확도를 `model_diagnostics` 에 함께 기록해야 함. 그래야 "B-1 적용 → 실사용 67%" 같은 비교가 가능해짐. (현재는 수동 실행)
