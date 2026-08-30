/// 분석 완료 후 본문 — 예측 카드·배너·가이드 조립.
library;

import 'dart:io';

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../data/confidence.dart';
import '../../../data/haptics.dart';
import '../../../data/image_quality.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import '../../../widgets/animated_entry.dart';
import '../../../widgets/feedback_card.dart';
import 'banners.dart';
import 'evidence_chips.dart';
import 'guide_card.dart';
import 'multi_material_card.dart';
import 'prediction_card.dart';

class ResultLoadedContent extends StatelessWidget {
  final File image;
  final Prediction prediction;
  final ImageQualityResult? quality;
  final PredictionWithRegions? regions;
  final bool objectsMulti;   // 물건 후보 ≥2 감지 (미선택 상태)
  final RegionInfo? regionInfo;  // 지역별 배출 규정 (설정 시)
  final bool regionSet;          // 지역 설정 여부 (규정 데이터가 없을 때 캡션 분기)
  final bool isSmartCapture;     // 다시 촬영하기 / 다시 선택하기 라벨
  final String? sceneNote;       // 장면 결과 vs 물건별 결과 불일치 안내
  const ResultLoadedContent({super.key, 
    required this.image,
    required this.prediction,
    this.quality,
    this.regions,
    this.objectsMulti = false,
    this.regionInfo,
    this.regionSet = false,
    this.isSmartCapture = false,
    this.sceneNote,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final assessment = assessConfidence(prediction);
    // reject: (1) 신뢰도 부족 (top1 < 0.55 또는 entropy > 0.7), 또는
    //         (2) 모델이 명시적으로 non_object 라고 분류 (폐기물 아님 — 재촬영 신호)
    //         → 둘 다 "기타/분류 불가" 로 정직하게 결론.
    final isNonObject = prediction.predictedClass == 'non_object';
    // 계층 응답의 reject(대분류조차 불확실) 도 동일하게 처리
    final hierReject = prediction.hier?.isReject ?? false;
    final reject = assessment.shouldReject || isNonObject || hierReject;
    // 계층 응답이면 롤업 조회 — 세부 비활성 시 부모 대분류 카드로 안내
    final info = reject
        ? infoFor('etc')
        : (prediction.hier != null
            ? infoForWithRollup(prediction.predictedClass,
                parentSlug: prediction.hier!.coarseClass)
            : infoFor(prediction.predictedClass));
    final accent = info?.color ?? cs.primary;

    final hasQualityIssue = quality?.hasIssue ?? false;
    final isMulti = regions?.isMultiMaterial ?? false;
    // realMulti — 진짜 다중재질로 인정하려면 두 조건 모두:
    //   1. global top1 이 확신 영역 (reject 아님)
    //   2. 모든 region 의 avg_conf >= 0.60 (region 별로도 확신)
    // 둘 중 하나라도 약하면 spurious multi (손바닥·마우스 같은 confident-wrong) →
    // 다중재질 카드 대신 PredictionCard 또는 reject 카드로 표시.
    // avg_conf 임계 0.75 — Fix 1.5 의 0.60 이 confident-wrong (손바닥·마우스) 통과시켜서 강화.
    // 진짜 다중재질 (PET+라벨 등) 은 보통 region 별 0.80+ 라 false negative 적음.
    final regionsHighConf = isMulti &&
        regions!.regions.every((r) => r.avgConf >= 0.75);
    final realMulti = regionsHighConf && !reject;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 사진 품질 문제 (어두움/흔들림) — 가장 먼저 안내
        if (hasQualityIssue) ...[
          QualityBanner(quality: quality!),
          const SizedBox(height: kSpaceM),
        ],

        // 메인 결론 — 세 갈래.
        if (realMulti) ...[
          // (1) 진짜 다중재질 — global 도 확신 + regions 도 2+ → "여러 재질" 메시지 + CAM 버튼.
          //     단일 클래스로 단정하지 않고, 아래 재질별 breakdown 이 실제 결과 역할.
          //     PredictionCard·GuideCard 는 이 케이스에서 숨김(다중재질 안내와 중복).
          AnimatedEntry(
            child: RejectCard(
              prediction: prediction,
              isMultiMaterial: true,
              image: image,
            ),
          ),
          const SizedBox(height: kSpaceM),
          AnimatedEntry(
            index: 1,
            child: MultiMaterialCard(regions: regions!.regions),
          ),
        ] else if (reject && topConfidentRegion(regions) != null) ...[
          // (2') 장면 단위론 불확실하지만 재질 영역 분석(CAM+검증 재분류)이
          //      확신하는 재질이 있는 경우 — 오버레이 배지와 결과 카드가
          //      어긋나던 불일치 해소: 영역 발견을 메인 답으로 승격.
          ...(() {
            final r = topConfidentRegion(regions)!;
            final rInfo = infoForWithRollup(r.slug, parentSlug: kFineToCoarse[r.slug]);
            final rAccent = rInfo?.color ?? cs.primary;
            return <Widget>[
              AnimatedEntry(
                child: RegionRescueBanner(region: r, info: rInfo),
              ),
              if (rInfo != null) ...[
                const SizedBox(height: kSpaceM),
                AnimatedEntry(
                  index: 1,
                  child: GuideCard(
                    info: rInfo,
                    accent: rAccent,
                    regionInfo: regionInfo,
                    regionSet: regionSet,
                    coarse: kFineToCoarse[r.slug] ?? r.slug,
                  ),
                ),
              ],
            ];
          })(),
        ] else if (reject && objectsMulti) ...[
          // (2) 여러 물건이 혼재해 장면 단위 확신이 분산된 경우 —
          //     "분류 불가" 로 단정하지 않고 물건별 분류(위 후보 카드·마커)로 안내.
          //     장면 reject 는 물건이 하나인데 어렵다는 뜻일 때만 의미가 있음.
          AnimatedEntry(
            child: RejectCard(prediction: prediction, isMultiObject: true),
          ),
        ] else if (reject) ...[
          // (3) 단일재질이지만 모델이 어느 클래스에도 확신 못 함 → etc reject.
          AnimatedEntry(child: RejectCard(prediction: prediction)),
          if (info != null) ...[
            const SizedBox(height: kSpaceM),
            AnimatedEntry(
              index: 2,
              child: GuideCard(
                info: info,
                accent: accent,
                regionInfo: regionInfo,
                regionSet: regionSet,
                coarse: prediction.hier?.coarseClass ?? prediction.predictedClass,
              ),
            ),
          ],
        ] else ...[
          // (3) 일반 분류 — 불확실 배너(있으면) → PredictionCard → 가이드.
          if (assessment.isUncertain) ...[
            AnimatedEntry(
              child: UncertainBanner(prediction: prediction),
            ),
            const SizedBox(height: kSpaceM),
          ],
          AnimatedEntry(
            index: assessment.isUncertain ? 1 : 0,
            child: PredictionCard(
              image: image,
              prediction: prediction, info: info, accent: accent,
              assessment: assessment,
            ),
          ),
          if (sceneNote != null) ...[
            const SizedBox(height: 8),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline,
                    size: 13, color: DsTokens.of(context).muted),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    sceneNote!,
                    style: TextStyle(
                        fontSize: 11.5,
                        height: 1.4,
                        color: DsTokens.of(context).muted),
                  ),
                ),
              ],
            ),
          ],
          if (info != null) ...[
            const SizedBox(height: kSpaceM),
            AnimatedEntry(
              index: 2,
              child: GuideCard(
                info: info,
                accent: accent,
                regionInfo: regionInfo,
                regionSet: regionSet,
                coarse: prediction.hier?.coarseClass ?? prediction.predictedClass,
              ),
            ),
          ],
        ],

        // 시맨틱 증거 배지 — 서버가 분류에 실제로 융합한 단서 노출 (신뢰 UI).
        // 분리배출 마크·라벨 문구·형태(정체) 인식 (SEMANTIC_FUSION_PLAN Phase 3)
        if (prediction.evidence.isNotEmpty) ...[
          const SizedBox(height: kSpaceM),
          AnimatedEntry(
            index: 2,
            child: EvidenceChips(evidence: prediction.evidence),
          ),
        ],


        const SizedBox(height: kSpaceM),

        // 피드백 — 결과가 정확했나요? (시안 16e)
        const SizedBox(height: 4),
        AnimatedEntry(
          index: 3,
          child: FeedbackCard(prediction: prediction, image: image),
        ),
        const SizedBox(height: 16),
        // 다시 촬영하기 — 결과 모달을 닫고 카메라(또는 갤러리 선택)로 복귀
        AnimatedEntry(
          index: 4,
          child: Material(
            color: DsTokens.of(context).surface,
            borderRadius: BorderRadius.circular(kRadiusMedium),
            child: InkWell(
              borderRadius: BorderRadius.circular(kRadiusMedium),
              onTap: () {
                Haptics.selection();
                Navigator.of(context).pop(false);
              },
              child: Container(
                height: 54,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: DsTokens.of(context).accentSoft,
                  ),
                  borderRadius: BorderRadius.circular(kRadiusMedium),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      isSmartCapture
                          ? Icons.photo_camera_outlined
                          : Icons.image_outlined,
                      size: 17,
                      color: DsTokens.of(context).accentDeep,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      isSmartCapture ? '다시 촬영하기' : '다른 사진 선택하기',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: DsTokens.of(context).accentDeep,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: kSpaceL),
      ],
    );
  }
}


/// reject 인데 재질 영역 분석이 확신하는 재질이 있으면 그 영역 반환.
/// (오버레이의 빗금 배지와 결과 카드 동기화 — 임계 0.6)
MaterialRegion? topConfidentRegion(PredictionWithRegions? regions) {
  final rs = regions?.regions;
  if (rs == null || rs.isEmpty) return null;
  final sorted = [...rs]..sort((a, b) => b.avgConf.compareTo(a.avgConf));
  final top = sorted.first;
  return top.avgConf >= 0.6 ? top : null;
}
