/// 온보딩 공통 프리미티브 — 주 버튼·단계 배지·시트 카드.
library;

import 'package:flutter/material.dart';

import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';

/// 한글 어절 단위 줄바꿈 — Flutter 는 한글을 글자 단위로 끊으므로("사시나/요?")
/// 어절 안의 글자 사이에 단어 결합자(U+2060)를 넣어 공백에서만 줄이 바뀌게 한다.
String keepAll(String text) =>
    text.split(' ').map((w) => w.split('').join('\u2060')).join(' ');

/// 주요 CTA — 52px · radius 16 · accent-700.
class OnboardingButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final VoidCallback? onTap;
  const OnboardingButton({super.key, required this.label, this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final enabled = onTap != null;
    return Material(
      color: enabled ? kAccent700 : t.border,
      borderRadius: BorderRadius.circular(kRadiusMedium),
      child: InkWell(
        borderRadius: BorderRadius.circular(kRadiusMedium),
        onTap: onTap,
        child: ConstrainedBox(
          constraints: const BoxConstraints(minHeight: 52),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                if (icon != null) ...[
                  Icon(icon, size: 18, color: enabled ? kNeutral100 : t.muted),
                  const SizedBox(width: 9),
                ],
                Flexible(
                  child: Text(
                    label,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: enabled ? kNeutral100 : t.muted,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}


/// 단계 표시 "1/3".
class StepBadge extends StatelessWidget {
  final String label; // '1/3'
  const StepBadge(this.label, {super.key});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final parts = label.split('/');
    return Text.rich(
      TextSpan(children: [
        TextSpan(text: parts.first),
        TextSpan(text: '/${parts.last}', style: TextStyle(color: t.faint)),
      ]),
      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: t.muted),
    );
  }
}


/// 시안의 바텀시트 카드 — 상단 radius 28 + 핸들.
class SheetCard extends StatelessWidget {
  final Widget child;
  const SheetCard({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Theme.of(context).scaffoldBackgroundColor,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [
          BoxShadow(
            color: kInkShadow.withValues(alpha: 0.3),
            offset: const Offset(0, -10),
            blurRadius: 34,
          ),
        ],
      ),
      padding: EdgeInsets.fromLTRB(
          24, 14, 24, 24 + MediaQuery.viewPaddingOf(context).bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color: t.handle,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          child,
        ],
      ),
    );
  }
}

// ─── ① 이용 동의 (17a) ─────────────────────────────────────────────────────
