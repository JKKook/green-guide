# 전자제품 인식 고도화 — v15 데이터 계획

작성 2026-10-18 · 상태: 초안(사용자 검토 대기) · 선행: v14 재학습 게이트 결과, PR #37(OOD 게이트 보정, 운영 반영 완료)

## 0. 한 줄 결론

운영 모델(v11, 7/21)이 전자제품을 못 알아보는 이유는 두 겹이었다 — ① OOD 게이트가 맞은 답을 "기타"로 버림(PR #37로 해결), ② **모델이 소형 전자기기를 학습한 적이 없음**(v11 학습 데이터의 전자제품 = AI-Hub 140 휴대폰 5,000장 + 기본 1,009장뿐). ②의 큰 몫은 v13에서 이미 수집한 Open Images 8,167장(노트북·모니터·태블릿·키보드·마우스 등 29품목)을 학습 중인 **v14가 흡수**한다. v15는 v14가 못 덮는 세 구멍 — *테스터가 실제로 찍는 품목(이어폰·충전기·케이블·보조배터리·스마트워치)*, *실사용 장면 도메인*, *전자제품 오탐을 부르는 음성 샘플* — 을 메우는 사이클이다.

## 1. 현황 (실측, 2026-10-18)

### 1-1. 운영 모델 v11 의 전자제품 인식력 — 테스트 분할 출처별 (프로덕션 ONNX, 출처당 25장)

| 출처 | electronics 판정률 | 주요 오답 |
|---|---|---|
| aihub385(휴대폰) | 100% | — |
| oi3_ipod | 92% | — |
| oi3_corded_phone | 64% | clothes 5 |
| oi3_tablet | 56% | paper 8 |
| oi3_monitor | 36% | paper 9 |
| oi3_laptop | 32% | paper 5 · paper_pack 4 |
| oi3_plug_socket / oi3_keyboard | 32% | paper 6~8 |
| oi2_camera | 28% | clothes 4 · metal 3 |
| oi2_mouse / oi2_keyboard | 20% | plastic · trash · paper |

→ v11 은 휴대폰 형태만 안다. 노트북·모니터·키보드는 **종이**로, 마우스·이어폰 케이스는 **플라스틱**으로 본다(테스터 맥북 사진 = plastic 0.97 "페트병").

### 1-2. v13/v14 학습 데이터의 전자제품 구성 (train 9,996 · val 2,141 · test 2,032)

- `electronics_boost` 13,167장 = AI-Hub 140 이동전화단말기 5,000 + Open Images 8,167
  - OI 품목(장): laptop 1,092 · monitor 724 · keyboard 723 · mouse 626 · tablet 488 · remote 420 · camera 343 · headphones 342 · ipod 297 · corded_phone 295 · television 294 · plug_socket 269 · coffeemaker 249 · printer 248 · kettle 247 · microwave 244 · fan 199 · calculator 190 · washing_machine 149 · blender 149 · refrigerator 148 · digital_clock 140 · alarm_clock 134 · flashlight 83 · toaster 51 · hair_dryer 23
  - v13 에서 실내 장면 대형가전 788장(television·fan·washing_machine·refrigerator 절반)을 `quarantine_v13_oi3_indoor/` 로 격리 — non_object 재현율 보호 목적
- 기본 `garbage-classification/electronics` 1,009장(aihub 1,000)
- v13(10-10, 게이트 FAIL 은 non_object 때문) 테스트 electronics recall **0.976 / precision 0.966**(n=2,032) → OI 품목을 학습하면 테스트에서는 잡힌다. v14(resnet50 + label smoothing 0.1, 운영 레시피)도 같은 데이터.

### 1-3. 테스터 실사용 사진에서 드러난 구멍 (10-12~10-18 업로드 63장)

- 찍힌 전자제품: 이어폰 케이스(3장), 맥북+맥미니(2장) 등 — **OI 에 없는 품목**(이어폰 케이스)과 **책상 위 복합 장면**(노트북+데스크톱+케이블)
- 오탐: 식당 금속 반찬 트레이 → electronics 0.91. 금속 식기·책 등 "전자제품처럼 생긴 비전자" 음성 샘플이 부족
- 사용자 확정 피드백 57건 중 electronics 7건 — 전부 `ai_training_opt_in=false` 라 **학습에 못 씀**

## 2. v15 범위

### 2-1. 추가 수집 — 전자제품 양성 (목표 +3,000)

| 경로 | 품목 | 수량 | 비고 |
|---|---|---|---|
| Open Images 4차 (`collect_openimages4.py`, 3차와 같은 사람-겹침 제외 규칙) | Headphones(인이어 비중↑) 300 · Watch 300(스마트워치만 — CLIP 제로샷 필터) · Mobile phone 300(케이스 낀 상태·손에 든 상태 우선) · Microphone 150 · Light switch 150 · Lamp 200 · Humidifier 150 · Scale 150 · Cassette deck 100 · Fax 100 | ~2,000 | 로컬 `ml/data/raw/openimages/`(이미지 21,244 · bbox CSV 2.2G) 에서 바로 채굴 가능 |
| 자체 촬영(팀, 폰 카메라) | 이어폰/이어폰 케이스 · 충전기/어댑터 · 케이블 · 보조배터리 · 스마트워치/밴드 · 블루투스 스피커 · 게임패드 · 공유기 · 전기면도기/전동칫솔 · 드라이기(현재 23장) | 품목당 30~50장 → ~400 | **OI 에 없는 품목은 이 경로뿐.** 책상·바닥·손·침대 등 배경 다양화, 2~3개 겹친 장면 포함. 20% 는 동결 테스트로 홀드아웃 |
| (선택) 공개 e-waste 데이터셋 | Kaggle/Roboflow e-waste 류 | 라이선스 확인 후 | CC-BY 급만. 대부분 제품 사진이라 도메인 효과는 제한적 |

### 2-2. 추가 수집 — 음성(하드 네거티브) (목표 +3,500, CLIP 재질 라우팅)

2차 채굴(`collect_openimages2.py`)의 **CLIP 제로샷 재질 라우팅**을 재사용해 다음 OI 클래스를 재질별 `*_boost` 로 넣는다. 불확실은 폐기.

| OI 클래스 | 기대 라우팅 | 수량 |
|---|---|---|
| Serving tray · Plate · Bowl · Tableware · Fork · Spoon · Frying pan · Wok · Mug · Coffee cup | metal_boost / glass / etc(도자기) | 1,500 |
| Book · Envelope · Poster · Paper towel · Picture frame | paper_boost | 800 |
| Box | cardboard_boost | 300 |
| Toy · Doll · Cosmetics · Perfume · Pen · Stapler · Scissors | plastic_boost / metal_boost / etc | 900 |

### 2-3. 도메인 보강

- OI 크롭 여백을 품목 크기의 ±30% 로 넓힌 "문맥 크롭" 변형을 소형 기기 6품목(laptop·keyboard·mouse·headphones·remote·tablet)에 한해 1,000장 추가 — 테스터 사진처럼 주변 사물이 함께 보이는 분포
- 실내 합성(`synthesize_indoor_scenes.py`)은 v9 배경 편향 전례가 있어 **쓰지 않는다**

### 2-4. 하지 않는 것

- 분류 체계 변경(electronics 세부 클래스 분할) — 안내문 차이(소형 수거함 vs 무상방문수거)를 위해 v16 후보로 미룸. 조건: 세부당 ≥1,500장
- Anthropic VLM 을 학습·라벨링에 쓰지 않음(사용자 방침: 사용자 2차 검증 용도만)
- 사용자 업로드 사진 사용 — opt-in 이 0건이라 불가

## 3. 성공 기준 (게이트)

| 항목 | 기준 | 측정 |
|---|---|---|
| 기존 게이트 | `retrain_hier.py` 게이트 PASS (대분류·non_object 재현율 무회귀) | 자동 |
| 동결 테스트 electronics | recall ≥ 0.97 · precision ≥ 0.96 (v13 수준 유지) | evaluation.json |
| **실사용 전자제품 세트**(신설) | 테스터 업로드 63장 중 전자제품 + 자체 촬영 홀드아웃 ~80장 → electronics 판정 ≥ 85% | `nonobj_by_source.py` 류 출처별 스크립트 |
| 오탐 | 음성 세트(금속 식기·책·장난감 홀드아웃 300장) → electronics 오판 ≤ 3% | 동일 |
| OOD | 새 ONNX 로 `build_hier_prototypes.py --onnx … ` 재생성, tau_soft/tau_hard 기록 — 실사용 전자제품 세트 하드 거부 0건 | eval_ood 류 |

## 4. 순서와 일정

1. **v14 게이트 확인** (진행 중 — 14:48 재시작, 머신 부하로 1.55 s/it, 이대로면 10-19 저녁) → PASS 면 프로토타입 재생성 + 출처별·테스터 세트 측정으로 §1-1 표를 v14 기준으로 갱신. 이 측정이 v15 수량을 조정한다(예: 노트북이 이미 90% 면 OI 추가 생략).
2. 채굴·촬영 병행 (v14 학습 중에도 가능 — **학습 사이클 도중 fine-staging 수정 금지** 규율 준수: `crops4/` 스테이징에만 쌓고 통합은 v14 종료 후)
3. 통합: phash 중복 제거, 동결 테스트 자동 제외(`synmo_` 와 같은 접두어 규칙 → `team_` 접두어는 20% 홀드아웃), `PER_CLASS_MAX` 확인
4. v15 학습 (운영 레시피 resnet50/smoothing 0.1, 머신 여유 시 ~11h) → 게이트 → 프로토타입 → HF Hub `serving/` 교체 + Space 재빌드
5. 병행 서버 작업(재학습 무관): CLIP identity 어휘에 소형 전자기기 추가 — 맥북·이어폰 케이스처럼 모델이 plastic 으로 미는 경우를 증거로 끌어당김

## 5. 리스크

- 머신 부하: 10-18 현재 메모리 31/32GB 사용·압축 7.9GB, sys CPU 71% — 학습 속도 3배 저하. 다른 세션·프로세스 정리 없이는 v14·v15 모두 느리다
- 자체 촬영은 사람 손이 든다(~2시간). 없으면 이어폰 케이스·충전기류는 v15 에서도 빠진다
- 음성 샘플 라우팅 오류(도자기 그릇을 metal 로)가 다른 클래스를 오염시킬 수 있음 → CLIP 확신 임계 보수적으로, 통합 전 100장 눈검수

## 부록 — 근거 산출물

- 운영 ONNX 출처별 판정: 세션 scratchpad `tasks/bucuwkqkv.output` (2026-10-18)
- 테스터 63장 전/후: scratchpad `eval_ood_before.json` / `eval_ood_after.json`
- v13 평가: `ml/classifier/outputs/failed_cycles/cnn_hier_20261010_054414/evaluation.json`
- OI 수집 스크립트: `ml/data/raw/openimages/collect_openimages{,2,3}.py`, 품목 매핑은 3차 `TARGETS`
