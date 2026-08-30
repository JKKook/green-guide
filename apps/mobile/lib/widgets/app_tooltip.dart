import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// 앱 공통 툴팁 — 모든 툴팁의 모양·동작을 한 곳에서 통일.
///
/// - 라운드 16(kRadiusMedium) · inverseSurface 배경 · 넉넉한 패딩
/// - 탭 트리거 (롱프레스 아님 — 발견 가능성 ↑)
/// - 본문 12.5/1.65 · [title]은 주아체 헤드라인 · [sections]의 용어는
///   inversePrimary 볼드로 강조
/// - [child]를 생략하면 관례인 ⓘ 아이콘이 붙는다
class AppTooltip extends StatelessWidget {
  /// 툴팁을 여는 대상. null 이면 ⓘ(info_outline) 아이콘.
  final Widget? child;
  final double iconSize;

  /// 단순 안내 본문.
  final String? message;

  /// 주아체 헤드라인 (선택).
  final String? title;

  /// (용어, 설명) 목록 — 용어는 강조색 볼드로 표시 (선택).
  final List<(String, String)>? sections;

  final Duration showDuration;

  const AppTooltip({
    super.key,
    this.child,
    this.iconSize = 17,
    this.message,
    this.title,
    this.sections,
    this.showDuration = const Duration(seconds: 8),
  }) : assert(message != null || sections != null,
            'message 또는 sections 중 하나는 필요합니다');

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    final children = <InlineSpan>[
      if (title != null)
        TextSpan(
          text: '$title\n',
          style: TextStyle(
            fontFamily: kDisplayFontFamily,
            fontWeight: FontWeight.w700,
            fontSize: 15.5,
            height: 2.0,
            color: cs.onInverseSurface,
          ),
        ),
      if (message != null) TextSpan(text: message),
      if (sections != null)
        for (var i = 0; i < sections!.length; i++) ...[
          TextSpan(
            text: '${i == 0 && message == null && title == null ? '' : '\n'}'
                '${sections![i].$1}\n',
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: cs.inversePrimary,
            ),
          ),
          TextSpan(
            text: '${sections![i].$2}${i == sections!.length - 1 ? '' : '\n'}',
          ),
        ],
    ];

    return Tooltip(
      richMessage: TextSpan(
        style: TextStyle(
          fontSize: 12.5,
          height: 1.65,
          color: cs.onInverseSurface,
        ),
        children: children,
      ),
      triggerMode: TooltipTriggerMode.tap,
      showDuration: showDuration,
      margin: const EdgeInsets.symmetric(horizontal: kSpaceL),
      padding: const EdgeInsets.all(kSpaceL),
      decoration: BoxDecoration(
        color: cs.inverseSurface.withValues(alpha: 0.97),
        borderRadius: BorderRadius.circular(kRadiusMedium),
      ),
      child: child ??
          Icon(Icons.info_outline, size: iconSize, color: cs.onSurfaceVariant),
    );
  }
}
