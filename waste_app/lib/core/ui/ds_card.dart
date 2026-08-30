/// 디자인 시스템 카드 — 화면마다 반복되던
/// `Container(decoration: BoxDecoration(color: t.surface, border: Border.all(color: t.border), borderRadius: ...))`
/// 의 단일 출처.
///
/// - 기본: surface 배경 + border 테두리 + 반경 16
/// - [elevated]: 시안 공통 그림자(0,1 / blur 2 / 14%)
/// - [tinted]: 강조 패널·칩 배경(accentChipBg + accent 테두리)
library;

import 'package:flutter/material.dart';

import '../../theme/design_tokens.dart';

/// 시안 카드 그림자 — 라이트/다크 공통.
const List<BoxShadow> kCardShadow = [
  BoxShadow(
    color: Color.fromRGBO(0x2B, 0x2B, 0x2D, 0.14),
    offset: Offset(0, 1),
    blurRadius: 2,
  ),
];

class DsCard extends StatelessWidget {
  const DsCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.radius = 16,
    this.elevated = false,
    this.tinted = false,
    this.clipBehavior = Clip.none,
    this.width,
    this.height,
    this.alignment,
    this.constraints,
  });

  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final double radius;
  final bool elevated;
  final bool tinted;
  final Clip clipBehavior;
  final double? width;
  final double? height;
  final AlignmentGeometry? alignment;
  final BoxConstraints? constraints;

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final borderColor = tinted ? (t.accentChipBorder) : t.border;
    return Container(
      width: width,
      height: height,
      margin: margin,
      padding: padding,
      alignment: alignment,
      constraints: constraints,
      clipBehavior: clipBehavior,
      decoration: BoxDecoration(
        color: tinted ? t.accentChipBg : t.surface,
        border: Border.all(color: borderColor),
        borderRadius: BorderRadius.circular(radius),
        boxShadow: elevated ? kCardShadow : null,
      ),
      child: child,
    );
  }
}
