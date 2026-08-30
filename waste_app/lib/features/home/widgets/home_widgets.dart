/// 홈 화면 구성 요소 — 헤더 아이콘 버튼·팁 카드·주간 스트립·힌트 카드.
library;

import 'package:flutter/material.dart';

import '../../../core/ui/ds_card.dart';
import '../../../data/collection_schedule.dart';
import '../../../data/tips.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';

/// 헤더 우측 38px 원형 아이콘 버튼.
class HeaderIconButton extends StatelessWidget {
  final DsTokens tokens;
  final VoidCallback onTap;
  final bool dimmed;
  final Widget child;

  /// 아이콘만 있는 버튼이라 스크린리더용 이름이 필요하다.
  final String semanticLabel;
  const HeaderIconButton({super.key, 
    required this.tokens,
    required this.onTap,
    this.dimmed = false,
    required this.semanticLabel,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: semanticLabel,
      child: Opacity(
      opacity: dimmed ? 0.6 : 1,
      child: Material(
        color: tokens.surface,
        shape: CircleBorder(side: BorderSide(color: tokens.border)),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onTap,
          child: SizedBox(
            width: 38,
            height: 38,
            child: Center(child: child),
          ),
        ),
      ),
      ),
    );
  }
}


/// 오늘의 팁 카드 — 블레이드 스트로크 패턴 배너 + 일별 팁.
class TipCard extends StatelessWidget {
  final DsTokens tokens;
  const TipCard({super.key, required this.tokens});

  @override
  Widget build(BuildContext context) {
    final tip = todayTip();

    return DsCard(
      elevated: true,
      radius: 24,
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 배너 — 13번 패턴 라이브러리 에셋, 팁이 바뀌는 날마다 교체
          SizedBox(
            height: 150,
            width: double.infinity,
            child: ColorFiltered(
              // 다크 모드 — 라이트 팔레트 배너를 살짝 가라앉혀 대비 유지
              colorFilter: tokens.dark
                  ? const ColorFilter.mode(Color(0xFF9AA3B0), BlendMode.modulate)
                  : const ColorFilter.mode(Colors.transparent, BlendMode.dst),
              child: Image.asset(
                todayTipBanner(),
                fit: BoxFit.cover,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '오늘의 팁',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 5),
                Text(
                  tip,
                  style: TextStyle(
                    fontSize: 12.5,
                    height: 1.55,
                    color: tokens.muted2,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}


/// 주간 수거 스트립 — 오늘부터 7일, 오늘 칩 강조.
class WeekStrip extends StatelessWidget {
  final DsTokens tokens;
  final int todayIdx;
  const WeekStrip({super.key, required this.tokens, required this.todayIdx});

  Color? _dotColor(PickupKind p) => switch (p) {
        PickupKind.plasticVinyl => kAccent500,
        PickupKind.paperBox => kAccent2400,
        PickupKind.general => tokens.faint,
        PickupKind.none => null,
      };

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        for (var i = 0; i < 7; i++) ...[
          if (i > 0) const SizedBox(width: 7),
          Expanded(
            child: Builder(builder: (context) {
              final dayIdx = (todayIdx + i) % 7;
              final isToday = i == 0;
              final dot = _dotColor(effectiveWeekSchedule()[dayIdx]);
              return Container(
                padding: const EdgeInsets.fromLTRB(0, 12, 0, 11),
                decoration: BoxDecoration(
                  color: isToday ? kAccent700 : tokens.surface,
                  border:
                      isToday ? null : Border.all(color: tokens.border),
                  borderRadius: BorderRadius.circular(kRadiusMedium),
                ),
                child: Column(
                  children: [
                    Text(
                      kDayNames[dayIdx],
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight:
                            isToday ? FontWeight.w700 : FontWeight.w600,
                        color: isToday ? kNeutral100 : tokens.muted,
                      ),
                    ),
                    const SizedBox(height: 7),
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isToday ? kNeutral100 : dot,
                      ),
                    ),
                  ],
                ),
              );
            }),
          ),
        ],
      ],
    );
  }
}


class HintCard extends StatelessWidget {
  final String currentApiUrl;
  final bool isTestMode;
  const HintCard({super.key, required this.currentApiUrl, required this.isTestMode});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final bgColor = isTestMode ? cs.primaryContainer : cs.surfaceContainerHigh;
    final fgColor = isTestMode ? cs.onPrimaryContainer : cs.onSurfaceVariant;
    return Container(
      padding: const EdgeInsets.all(kSpaceM),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(kRadiusMedium),
      ),
      child: Row(
        children: [
          Icon(
            isTestMode ? Icons.usb : Icons.cloud_outlined,
            size: 18,
            color: fgColor,
          ),
          const SizedBox(width: kSpaceS),
          Expanded(
            child: Text(
              isTestMode
                  ? '테스트 모드 — adb reverse 로 PC API 연결'
                  : 'API: $currentApiUrl',
              style: TextStyle(fontSize: 12, color: fgColor),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
