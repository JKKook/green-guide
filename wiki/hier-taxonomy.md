# 계층 Taxonomy — 대분류 14 × 세부 25

> 소스: `waste-classifier/src/taxonomy.py` (단일 진실), `waste-api/models/taxonomy.json`, `docs/plans/GREENGUIDE_BLUEPRINT.md` §1 (2026-08-13 탐색)

flat softmax 확장의 한계(클래스↑=클래스당 데이터↓, "애매하면 대분류만 답하기" 불가)를 극복하는 2단 계층. 모델은 **fine 25 단일 head**만 학습하고 대분류는 결정적 롤업 `P(coarse) = Σ P(fine children)`.

| coarse (14) | fine children (25) |
|---|---|
| paper | paper_other, cardboard |
| paper_pack | carton, paper_cup |
| glass | glass_brown, glass_green, glass_clear, glass_deposit, glass_etc |
| metal | metal |
| plastic | pet, plastic_other |
| vinyl | vinyl_clean, vinyl_dirty |
| styrofoam | styrofoam_white, styrofoam_color, styrofoam_dirty |
| clothes | clothes |
| food_waste | food_waste |
| electronics | electronics |
| hazardous | battery (향후 fluorescent, medicine) |
| trash | trash_other, light_bulb (⚠️ 전구≠형광등 → 일반쓰레기) |
| etc | etc (내부 신호, 비노출) |
| non_object | non_object (손·신체·배경) |

⚠️ BLUEPRINT는 대분류 12+내부 2로 설계했으나 구현은 14×25 — 숫자 차이 있음 ([planning-docs](planning-docs.md) 모순 목록).

## 규칙

- 모든 fine은 정확히 하나의 parent로 롤업. **배출법이 대분류와 같으면 세부를 만들지 않는다**(분류를 위한 분류 금지).
- **데이터-게이트 활성화**: fine이 active=true 되려면 train ≥300, frozen ≥30, 부모 recall 회귀 없음, f1 ≥0.80 (실제 판정: `support≥30 AND (f1≥0.80 OR guidance_safe_f1≥0.85)`).
- `GUIDANCE_GROUPS`: 배출 안내가 동일한 형제 = {carton, paper_cup}(종이팩 수거함), {glass_brown/green/clear/etc}(유리병 수거함). **glass_deposit(보증금 반환)은 안내가 달라 제외.** guidance-safe f1 지표의 근거.
- `LEGACY_LABEL_SUPERVISION`: 구 manifest 13라벨 → fine 또는 coarse 감독(예: glass→coarse, paper→fine paper_other). 세부 미상 데이터 2.3만 장을 coarse 감독으로 재활용.
- `STAGING_DIR_SUPERVISION`: fine-staging 폴더명 → 감독 매핑 (`*_boost`는 coarse, taco 계열 등).

## 서빙 신뢰도 게이트

`taxonomy.json` 사이드카로 모델과 함께 배포: fine ≥0.60 & margin ≥0.15 → 세부 안내 / coarse ≥0.55 → 대분류만 / 미달 → reject. 온디바이스([waste-app](waste-app.md))도 동일 게이트·롤업 적용.

관련: [hier-training-pipeline](hier-training-pipeline.md) · [ood-openset](ood-openset.md)
