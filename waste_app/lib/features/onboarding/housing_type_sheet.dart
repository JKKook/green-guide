/// 주거 형태 선택 시트 — 온보딩 ③ · 설정에서 공용.
library;

import 'package:flutter/material.dart';

import '../../data/haptics.dart';
import '../../data/settings_store.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import 'widgets/onboarding_primitives.dart';

/// 주거 형태 선택 시트. 온보딩(2/3 배지)·설정(배지 없음) 공용.
Future<HousingType?> showHousingTypeSheet(
  BuildContext context, {
  HousingType? current,
  String? stepLabel,
}) {
  return showModalBottomSheet<HousingType>(
    context: context,
    isDismissible: stepLabel == null,
    enableDrag: stepLabel == null,
    builder: (_) => _HousingTypeSheet(current: current, stepLabel: stepLabel),
  );
}


class _HousingTypeSheet extends StatefulWidget {
  final HousingType? current;
  final String? stepLabel;
  const _HousingTypeSheet({this.current, this.stepLabel});

  @override
  State<_HousingTypeSheet> createState() => _HousingTypeSheetState();
}


class _HousingTypeSheetState extends State<_HousingTypeSheet> {
  late HousingType _value = widget.current ?? HousingType.house;

  Widget _option({
    required HousingType type,
    required IconData icon,
    required String title,
    required String subtitle,
  }) {
    final t = DsTokens.of(context);
    final selected = _value == type;
    return InkWell(
      borderRadius: BorderRadius.circular(20),
      onTap: () {
        Haptics.selection();
        setState(() => _value = type);
      },
      child: Container(
        padding: const EdgeInsets.all(kSpaceL),
        decoration: BoxDecoration(
          color: selected ? t.accentChipBg : t.surface,
          border: Border.all(
            color: selected ? kAccent500 : t.border,
            width: selected ? 1.5 : 1,
          ),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Container(
              width: 46,
              height: 46,
              decoration: BoxDecoration(
                color: selected ? t.bannerBg : t.surface,
                border: Border.all(
                  color: selected
                      ? (t.accentChipBorder)
                      : t.border,
                ),
                borderRadius: BorderRadius.circular(kRadiusMedium),
              ),
              child: Icon(icon,
                  size: 22, color: selected ? t.accentChipText : t.muted2),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: const TextStyle(
                          fontSize: 15.5, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 2),
                  Text(subtitle,
                      style: TextStyle(fontSize: 12, color: t.muted2)),
                ],
              ),
            ),
            Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                color: selected ? kAccent700 : Colors.transparent,
                border: selected
                    ? null
                    : Border.all(
                        color: t.handle,
                        width: 1.5),
                shape: BoxShape.circle,
              ),
              child: selected
                  ? const Icon(Icons.check, size: 12, color: kNeutral100)
                  : null,
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return SheetCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Expanded(
                child: Text(
                  '어떤 집에 사세요?',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                ),
              ),
              if (widget.stepLabel != null) StepBadge(widget.stepLabel!),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '주거 형태에 따라 분리배출 방법이 달라져요',
            style: TextStyle(fontSize: 12.5, color: t.muted2),
          ),
          const SizedBox(height: 18),
          _option(
            type: HousingType.apartment,
            icon: Icons.apartment_outlined,
            title: '아파트 · 오피스텔',
            subtitle: '단지 내 분리배출장에 상시 배출',
          ),
          const SizedBox(height: 10),
          _option(
            type: HousingType.house,
            icon: Icons.home_outlined,
            title: '주택 · 빌라',
            subtitle: '동네 수거 요일에 맞춰 문 앞 배출',
          ),
          const SizedBox(height: 16),
          OnboardingButton(
            label: '선택 완료',
            onTap: () => Navigator.of(context).pop(_value),
          ),
          const SizedBox(height: 12),
          Center(
            child: Text(
              '나중에 설정에서 변경할 수 있어요',
              style: TextStyle(fontSize: 11.5, color: t.muted),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── ④ 주택·빌라 — 분리 수거 설정 (17d) ──────────────────────────────────────
