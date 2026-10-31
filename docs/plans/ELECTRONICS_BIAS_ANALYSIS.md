# electronics 과대예측·실사용 정확도 격차 원인 분석 (2026-10-31, 읽기 전용 진단)

저장소 develop @4e1a5bc. 파일 수정·학습 없음. 진단 추론만 실행(리더 세션 스크래치 `diag_infer.py`·`imgstats.py`, 저장소 미반영). 리더 보충: 로컬 피드백 `local_feedback/meta.jsonl` 195행 중 predicted_class=electronics 53행(27%) — 보고서 끝 "미확인" 항목은 이 파일로 재현됨.

## 0. 모델 정체 정리 (질문의 수치가 두 모델에 걸쳐 있음)
- **0.9315** = v11(`v20260715_124216`) fine 정확도(`outputs/logs/diagnosis/hier_history.jsonl`), 운영 등록 `20260805_152341`(`docs/plans/PRODUCTION_READINESS.md:74`). 10-21 전까지 서빙.
- **val_fine 0.90 / val_coarse 0.94** = v14(`v20261020_100742`, `outputs/models/cnn_hier/taxonomy.json`: 0.8946/0.9352). test 대분류 **0.9566**, fine 0.9236(`logs/cnn_hier/evaluation.json`). 10-21 HF `serving/` 교체·Space 재부팅(`logs/tester_eval/deploy_serving.py`, tau_hard p94).
- 즉 "전자제품 데이터 추가 후 오분류 증가"의 before=v11, after=v14. 아래 모두 그 대비.

## A. electronics 클래스의 추가·확장 이력
| 시점 | 내용 | 근거 |
|---|---|---|
| 7/10 | AI-Hub 140 이동전화단말기 5,000장 → `fine-staging/electronics_boost/aihub385_clean__aihub140_*` | 파일 mtime, `wiki/dataset-staging.md:19` |
| 7/15 | **v11** 학습. electronics train = boost 3,574 + 기본 589 = **4,163 / 139,063 = 3.0%** (출처 aihub385 3,574·aihub 582) | `backups/cnn_hier_20261010_054857/hier_splits.json`(7/15 생성) |
| 7/20 | Open Images 2차 `oi2_*` 2,237장(blender·camera·mouse·keyboard…) 추가 | `ml/data/raw/openimages/collect_openimages2.py` mtime, 파일 mtime 7/20 11:37 |
| 7/21 | 재학습 시도 → failed_cycles(미배포). train electronics oi2 1,793 포함 | `failed_cycles/cnn_hier_20260721_080243/hier_splits.json` |
| 10/05 | OI 3차 `oi3_*` **5,930장**(29품목: laptop 1,092·monitor 724·keyboard 723·mouse 626·tablet 488·remote 420·camera·headphones·television·plug_socket·coffeemaker·printer·kettle·microwave·fan·calculator·washing_machine·blender·refrigerator·clock·flashlight·toaster·hair_dryer) | `collect_openimages3.py`(10-05 12:21), `docs/plans/ELECTRONICS_V15_DATA_PLAN.md` §1-2 |
| 10/05·10/10 | v13(resnet18) 2회 게이트 FAIL(non_object recall 0.87). 10-10 실내 대형가전 788장 `quarantine_v13_oi3_indoor/` 격리 | `failed_cycles/cnn_hier_2026100{5,10}_*/evaluation.json`, `data/raw/quarantine_v13_oi3_indoor` |
| 10/18~20 | **v14**(resnet50, label_smooth 0.1) PASS. electronics train = boost 9,466 + 기본 530 = **9,996 / 153,594 = 6.5%**, fine 감독 중 2위(metal 14,599 다음). 출처 oi3 4,505·aihub385 3,284·oi2 1,677·aihub 528 | `data/splits/hier_splits.json`, `logs/v14_cycle.log:23` |

- 데이터 출처: AI-Hub(140 휴대폰 = 스튜디오 제품컷)·Open Images bbox 크롭(자연 장면, 여백 12%, `collect_openimages3.py:147-150`)·자체 촬영 0·합성 0. 사용자 피드백 electronics 7건은 전부 opt-in=false 로 미사용(V15 계획 §1-3).
- 손실·가중: `HierarchicalLoss` fine CE(label_smoothing 0.1) + coarse 롤업 NLL, **inverse-freq 가중치 median×4 상한**(`hier_train.py:80-166`, `train.py:35-48`, `config.py:101`). 샘플러 없음(단순 셔플). electronics 는 대형 클래스라 가중치가 **가장 낮은 축** → 가중치가 편향을 만든 게 아니라 **자연 장면 이미지의 물량**이 만든 것.
- 전처리/증강: 전체 이미지를 224×224 **통째 리사이즈**(crop 없음, `dataset.py:46`), flip·grayscale 20%·color jitter 만(`dataset.py:109-135`). **RandomResizedCrop·배경 증강 없음.**

### 학습 이미지 특성 실측 (표본 120장/군, `imgstats.py`; border_L = 가장자리 평균 밝기 0~255)
| 군 | n | 해상도 p50 | border_L | border_std | 채도 |
|---|---|---|---|---|---|
| electronics oi3 | 5,930 | 470×454 | **99** | 51 | 71 |
| electronics oi2 | 2,237 | 398×357 | **107** | 47 | 62 |
| electronics aihub140(휴대폰) | 5,000 | 256×191 | 148 | 21 | 56 |
| plastic(aihub) | 10,274 | 256×256 | 65 | 43 | 55 |
| plastic_boost(aihub385_dirty) | 3,590 | 220×253 | **30** | 21 | 64 |
| metal(aihub) | 10,155 | 256×256 | 54 | 55 | 105 |
| metal_boost | 10,718 | 222×252 | **37** | 29 | 80 |
| pet(aihub385) | 13,923 | 204×224 | **21** | 19 | 68 |
| non_object | 1,787 | 256×256 | 124 | 41 | 54 |
| **테스터 실사용 93장** | 93 | 386×640(세로 91%) | **106** | **53** | 46 |

→ 재활용품(plastic·metal·pet)은 AI-Hub 분류기 박스의 **어두운 배경**(21~65), OI electronics 만 **밝은 자연 장면**(99~107). 실사용 사진(106/53)은 OI electronics 분포와 겹치고 플라스틱·금속 학습 분포와는 전혀 겹치지 않는다.

## B. test_acc 의 정체와 실사용 평가 기록
- test = 같은 폴더의 stratified 랜덤 분할 + 동결(`hier_dataset.py:122-135`, `hier_frozen_test.json` 18,206장 중 electronics_boost 1,635). electronics test 2,032 = 전부 aihub140/oi2/oi3 → **동일 분포**. `by_source` 에 realworld n=5 뿐.
- v14 test electronics P 0.992 / R 0.978. 혼동행렬 electronics 열(타→electronics): paper 5·metal 3·trash 2·glass 1·plastic 1·non_object 4 — 테스트에서는 과대예측이 **보이지 않음**(어두운 배경 재활용품은 electronics 로 안 감).
- 실사용 평가는 있었고 수치는 기록됨: `scripts/realworld_eval_hier.py`(순수 ONNX, 마스킹·TTA 없음) 피드백 57건 **대분류 38.6%(22/57)**, plastic 3/13, metal 1/6, etc 0/12 (`logs/cnn_hier/realworld_eval.json`, `v14_cycle.log` 말미). 오류 36건 중 pred=electronics 13(plastic→6, metal→5, etc→1, food→1), pred=non_object 8. **게이트 조건이 아님**(KPI 85% 메모만).
- 테스터 93장 전후(`logs/tester_eval/v14_visual_notes.md`, eval_before/after.json): electronics 표시 **13→44**, coarse argmax electronics 13→55, plastic 20→0, glass 10→0. 신규 오류: 자동차·새우요리·랜덤노이즈→electronics.
- 골든셋 AI 검토(`PRODUCTION_READINESS.md:61`): 불일치 19건 중 12건이 플라스틱 용기류→electronics.

## C. 실사용 vs 학습 도메인 차이
- 서빙 전처리 = 학습과 동일: EXIF 정립 → 전체 프레임 224×224 bilinear 통째 리사이즈 → ImageNet 정규화(`services/inference-api/src/preprocess.py:72`, `dataset.py:46`). **중앙 crop 은 없다** → 물체를 잃지 않지만 720×1280 세로 사진이 가로 3.2×·세로 5.7× 비율로 **찌그러지며** 배경이 프레임 대부분을 차지.
- 앱 "가이드 프레임 크롭"은 뷰파인더 가시 영역 − 14dp inset(`apps/mobile/lib/data/image_prep.dart:140-163`, `live_camera_screen.dart:25,185`) = 사실상 **전체 화면 세로 프레임**. 주석의 "bbox 밀착 크롭과 정렬"은 성립하지 않음. 긴 변 1600px·quality 90 재인코딩(`gallery_confirm_screen.dart:32-33`, `capture_entry_sheet.dart:37-38`)은 해상도 측면 무해.
- 학습 AI-Hub 크롭은 물체가 프레임을 채우는 256² bbox 크롭, OI 는 bbox+12% 여백 → 실사용의 "작은 물체 + 넓은 밝은 배경(책상·바닥·손)" 구성은 **OI electronics 만 가진 구성**.

## D. 서빙 측 증폭 요인 (93장 진단 추론, v14 vs v11)
| 단계 | v11 electronics | v14 electronics | 근거 |
|---|---|---|---|
| 순수 argmax(마스킹 없음) | 18/93 | **51/93** (non_object 20, plastic 0) | diag_infer |
| non_object 마스킹(logit −1e9 → softmax 재분배, `hier_inference.py:184-186`) | 18 | **57** (+6; non_object→trash 12) | diag_infer |
| 회전 TTA 최고확신 선택(`predict_rotations`) | 18 | 55 | diag_infer·eval_after |
| 게이트 coarse≥0.55(`taxonomy.json gate`) | — | electronics conf p10 0.34 · p50 0.73, <0.55 는 14/57 → **43장 표시** | eval_after: fine 39+coarse 5 |
- 합성 프로브: 균일 배경(40·128·220·255·나무색) → v14 non_object 0.93~0.97(정상), **랜덤 노이즈 → electronics 0.76**(v11 은 glass). 마스킹 뒤 균일 배경은 trash/hazardous 0.16~0.24 로 재분배.
- OOD: v14 prototype 이 OI 자연사진을 포함해 실사용 평균 거리 0.351→0.418 로 커졌지만 tau_hard 0.672→0.769(p97.5)→0.674(p94) 로 같이 완화 → 하드 거부 0건, 노이즈 d=0.704 통과(`v14_visual_notes.md`). `ood_relax` 는 탭 경로만(`routers/inference.py:154`).
- prior 융합: OCR 은 fine_conf<0.75 일 때만, CLIP·CAM 은 **탭 경로만**(`routers/inference.py:176-202`) → 장면 사진의 electronics 오답을 교정할 증거 단계가 없다. CLIP 어휘에 electronics 9컨셉 존재(`scripts/build_clip_concepts.py:91-99`).
- 결론: 증폭은 +6~12% 수준, **주범은 모델 자체(51/93)**.

## E. 원인 가설(중요도순) · 검증 · 수정 후보
| # | 가설 | 핵심 근거 | 지금 검증 | 수정(재학습 無) | 수정(재학습 有) |
|---|---|---|---|---|---|
| 1 | **배경·장면 혼동**: OI electronics 6,182장이 유일한 "밝은 자연 장면" 양성 → "자연 사진 = electronics" 학습 | 표 A(border_L 99~107 vs 21~65, 실사용 106), v11→v14 18→51, plastic 5→0, 노이즈→elec | `diag_infer.py` 를 골든 51장에 실행; 물체 영역을 어두운 배경에 합성한 변형으로 electronics 비율 비교. CAM(`/predict-hier` want_cam)으로 electronics 오답의 활성이 배경에 있는지 | electronics logit bias/temperature 보정(골든 51+테스터 93 홀드아웃 분리, 반나절) | **하드 네거티브**: plastic/metal/glass 를 OI 자연장면에서 CLIP 재질 라우팅으로 +3,500(V15 계획 §2-2) + 배경 증강·RandomResizedCrop; ~11h 학습 + 수집 1~2일 |
| 2 | **프레이밍 불일치**: 학습=물체 밀착 크롭, 실사용=전체 세로 프레임(가이드 크롭이 사실상 전체) 통째 리사이즈 | C절 | 골든에 `image_io.auto_crop_to_object`(u2netp, 코드 존재) 적용 후 재분류 비교 | `/predict-hier` 장면 경로에 auto_crop 적용(+0.3~1s), 앱 가이드 프레임 축소 | RandomResizedCrop(scale 0.3~1)·여백 랜덤화 증강 |
| 3 | **non_object 마스킹·OOD 완화·게이트**가 2순위 답(electronics)을 노출 | D절(+6, tau 완화, p50 0.73) | diag_infer 결과 그대로(마스킹 off 시 non_object 20) | raw non_object≥0.5 면 재분배 대신 coarse-only/reject; electronics 전용 coarse 게이트 0.7; tau_hard v11 수준(0.67) 복귀 — 몇 시간 | — |
| 4 | **electronics 의미 범위 과확장**: 믹서·전기포트·토스터·손전등·계산기·세탁기 등 플라스틱/금속 외장 가전 → "플라스틱/금속 하우징 물체 = electronics" | 골든 12/19 플라스틱 용기→elec, OI 품목표 | 골든 electronics 오답 상위 fine_top5 에 어떤 OI 품목 유사성이 있는지 CLIP identity 로 조회(탭 경로 코드 재사용) | 장면 경로에도 CLIP identity 적용해 electronics 컨셉 확신<0.3 이면 electronics 억제(1~2일, +0.2s) | OI 품목을 테스터 범위(노트북·마우스·키보드·이어폰·충전기)로 축소, 가전은 별도 세부로 |
| 5 | **평가 맹점**: test 가 동일 출처 분할, 실사용 38.6% 는 게이트 아님 | B절 | `realworld_eval_hier.py` 이미 돌아감(57건) | 골든 51·테스터 93 을 게이트로 승격(electronics 오탐률 ≤3%, V15 §3) | 자체 촬영 400장 중 20% 홀드아웃 |

- 재학습 없는 조합(1 보정 + 2 auto_crop + 3 게이트/마스킹)은 1~3일, 실사용 분포 전환 없이 **증상 완화**. 근본 해결은 가설 1·2 의 재학습(네거티브 + 크롭/배경 증강) — V15 계획 §2-2·§2-3 과 일치하되 **"문맥 크롭 1,000장 추가"보다 재활용품 쪽 자연장면 네거티브가 우선**이어야 함(지금은 electronics 만 문맥을 가짐).
- 미확인: 로컬 피드백 195장의 27% 수치는 저장소 내 산출물에서 재현 못 함(`local_feedback/item_candidates.jsonl` 9행뿐). golden_v2 확신 분포는 리더 담당.
