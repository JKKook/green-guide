import 'package:flutter/material.dart';

import '../api/models.dart';
import '../core/di/app_scope.dart';
import '../data/haptics.dart';
import '../data/waste_info.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';

/// 밝은 재질 컬러는 아이콘 대비가 낮아 블루그레이로 보정.
Color wasteIconColor(WasteInfo info) => info.color.computeLuminance() > 0.6
    ? const Color(0xFF607D8B)
    : info.color;

/// 품목 배출 기준 바텀시트 — 통합 검색·기록 공용.
/// 지역이 설정돼 있으면 결과 화면과 같은 우리 동네 조례 규정을 함께 보여준다.
void showCriteriaSheet(BuildContext context, WasteInfo info) {
  Haptics.selection();
  showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (_) => _CriteriaSheetBody(info: info),
  );
}

class _CriteriaSheetBody extends StatefulWidget {
  final WasteInfo info;
  const _CriteriaSheetBody({required this.info});

  @override
  State<_CriteriaSheetBody> createState() => _CriteriaSheetBodyState();
}

class _CriteriaSheetBodyState extends State<_CriteriaSheetBody> {
  late final Future<RegionInfo?> _regionInfo = _loadRegionInfo();

  /// 설정된 지역의 배출 규정 — 미설정·오프라인이면 null (전국 공통만 표시).
  Future<RegionInfo?> _loadRegionInfo() async {
    try {
      final region = await AppScope.settings.getRegion();
      if (region == null) return null;
      final client = await AppScope.api();
      return await client.fetchRegionInfo(region.$1, region.$2);
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final info = widget.info;
    final cs = Theme.of(context).colorScheme;
    final t = DsTokens.of(context);
    // 세부품목은 부모 대분류 기준으로 재활용/음식물/일반 규정을 고른다.
    final coarse = info.level >= 2 && info.parentSlug != null
        ? info.parentSlug!
        : info.classKey;

    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(kSpaceXL, 0, kSpaceXL, kSpaceXL),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFFEDEEED)
                        : cs.surfaceContainerHigh,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(info.icon, color: wasteIconColor(info), size: 23),
                ),
                const SizedBox(width: kSpaceM),
                Expanded(
                  child: Text(
                    info.displayName,
                    style: Theme.of(context)
                        .textTheme
                        .titleLarge
                        ?.copyWith(fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
            const SizedBox(height: kSpaceM),
            Text(
              info.summary,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            // 우리 동네 규정 — 결과 화면 가이드 카드와 같은 행 (공공데이터, 조례 기준)
            FutureBuilder<RegionInfo?>(
              future: _regionInfo,
              builder: (context, snap) {
                final region = snap.data;
                final rows = region?.representative?.rowsFor(coarse) ??
                    const <(String, String)>[];
                if (region == null || rows.isEmpty) {
                  return const SizedBox.shrink();
                }
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: kSpaceM),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 9, vertical: 4),
                      decoration: BoxDecoration(
                        color: t.accentChipBg,
                        border: Border.all(color: t.accentChipBorder),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.place_outlined,
                              size: 11, color: t.accentChipText),
                          const SizedBox(width: 4),
                          Text(
                            '${region.sigungu} 조례 기준',
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: t.accentChipText,
                            ),
                          ),
                        ],
                      ),
                    ),
                    for (final (label, value) in rows) ...[
                      const SizedBox(height: kSpaceS),
                      _CriteriaRow(label: label, value: value),
                    ],
                  ],
                );
              },
            ),
            if (info.bin.isNotEmpty) ...[
              const SizedBox(height: kSpaceM),
              _CriteriaRow(label: '배출함', value: info.bin),
            ],
            if (info.howTo.isNotEmpty) ...[
              const SizedBox(height: kSpaceS),
              _CriteriaRow(label: '배출방법', value: info.howTo.join('\n')),
            ],
            const SizedBox(height: kSpaceS),
            Text(
              info.trainedInModel
                  ? '실제 판정은 사진 속 재질 분석 결과를 따릅니다'
                  : 'AI 분류 대상은 아니지만 배출 방법을 안내해 드려요',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: cs.onSurfaceVariant,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CriteriaRow extends StatelessWidget {
  final String label;
  final String value;
  const _CriteriaRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 64,
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant,
                  fontWeight: FontWeight.w700,
                ),
          ),
        ),
        Expanded(
          child: Text(value, style: Theme.of(context).textTheme.bodyMedium),
        ),
      ],
    );
  }
}
