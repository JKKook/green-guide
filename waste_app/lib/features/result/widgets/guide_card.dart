/// 배출 가이드 카드 (방법·주의·수거함).
library;

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../core/ui/ds_card.dart';
import '../../../data/waste_info.dart';
import '../../../theme/design_tokens.dart';

/// 온디바이스 신뢰도가 낮아서 cloud 가 재검증한 경우 표시되는 작은 배지.
/// 이렇게 버려요 — 전국 공통 요령 + (지역 설정 시) 우리 동네 조례 규정 통합.
class GuideCard extends StatelessWidget {
  final WasteInfo info;
  final Color accent;
  final RegionInfo? regionInfo;
  final bool regionSet;
  final String coarse;
  const GuideCard({super.key, 
    required this.info,
    required this.accent,
    required this.regionInfo,
    required this.regionSet,
    required this.coarse,
  });

  static const _recyclables = {
    'paper', 'paper_pack', 'glass', 'metal', 'plastic', 'vinyl',
    'styrofoam', 'clothes',
  };

  /// 지역 규정 행 — 재질을 재활용/음식물/일반으로 매핑해 해당 분류의 규정 추출.
  List<(String, String)> _regionRows(RegionRule r) {
    final String? method;
    final String? days;
    if (coarse == 'food_waste') {
      method = r.methodFood;
      days = r.daysFood;
    } else if (_recyclables.contains(coarse)) {
      method = r.methodRecycle;
      days = r.daysRecycle;
    } else {
      method = r.methodGeneral;
      days = r.daysGeneral;
    }
    return [
      if (method != null && method.isNotEmpty) ('배출 방법', method),
      if (days != null && days.isNotEmpty) ('배출 요일', days),
      if (r.emitTime != null && r.emitTime!.isNotEmpty) ('배출 시간', r.emitTime!),
      if (r.noCollectDay != null && r.noCollectDay!.isNotEmpty)
        ('미수거일', r.noCollectDay!),
    ];
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final rule = regionInfo?.representative;
    final regionRows = rule == null ? const <(String, String)>[] : _regionRows(rule);
    final hasRegion = regionRows.isNotEmpty;
    final steps = [
      if (info.howTo.isEmpty && info.bin.isNotEmpty) info.bin,
      ...info.howTo.take(3),
    ];
    final bodyColor = t.body;

    return DsCard(
      radius: 20,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Expanded(
                child: Text(
                  '이렇게 버려요',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
                ),
              ),
              if (hasRegion)
                // 지역 조례 기준 배지
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 9, vertical: 4),
                  decoration: BoxDecoration(
                    color: t.accentChipBg,
                    border: Border.all(
                      color: t.accentChipBorder,
                    ),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.place_outlined,
                          size: 11, color: t.accentChipText),
                      const SizedBox(width: 4),
                      Text(
                        '${regionInfo!.sigungu} 조례 기준',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: t.accentChipText,
                        ),
                      ),
                    ],
                  ),
                )
              else if (info.bin.isNotEmpty && info.howTo.isNotEmpty)
                Flexible(
                  child: Text(
                    info.bin,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 11.5, color: t.muted),
                  ),
                ),
            ],
          ),
          // 우리 동네 규정 — 배출 방법·요일·시간·미수거일 (공공데이터, 조례 기준)
          if (hasRegion) ...[
            const SizedBox(height: 12),
            for (final (i, (label, value)) in regionRows.indexed) ...[
              if (i > 0) const SizedBox(height: 6),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    width: 64,
                    child: Text(
                      label,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: t.accentChipText,
                      ),
                    ),
                  ),
                  Expanded(
                    child: Text(
                      value,
                      style: TextStyle(
                          fontSize: 13, height: 1.5, color: bodyColor),
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 12),
            Container(height: 1, color: t.border),
            const SizedBox(height: 12),
            Text(
              '분리배출 요령',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.44,
                color: t.muted,
              ),
            ),
            const SizedBox(height: 8),
          ] else
            const SizedBox(height: 12),
          // 전국 공통 분리배출 요령
          for (final (i, s) in steps.indexed) ...[
            if (i > 0) const SizedBox(height: 10),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 1),
                  child: Icon(Icons.check, size: 16, color: t.accentStrong),
                ),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(
                    s,
                    style: TextStyle(
                        fontSize: 13, height: 1.5, color: bodyColor),
                  ),
                ),
              ],
            ),
          ],
          const SizedBox(height: 12),
          Text(
            hasRegion
                ? '출처 · ${regionInfo!.sido} ${regionInfo!.sigungu} 폐기물 관리 조례 '
                    '(행안부 생활쓰레기 배출정보 표준데이터) · 관리구역에 따라 다를 수 있어요'
                : regionSet
                    ? '전국 공통 안내 · 우리 동네 조례 데이터는 아직 준비 중이에요'
                    : '전국 공통 안내 · 지역을 설정하면 우리 동네 조례 기준도 함께 알려드려요',
            style: TextStyle(fontSize: 10.5, height: 1.4, color: t.muted),
          ),
        ],
      ),
    );
  }
}
