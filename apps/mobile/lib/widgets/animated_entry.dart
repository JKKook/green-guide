/// 모션 헬퍼: 스태거 페이드+슬라이드 진입 효과.
library;

import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// 자식이 마운트되는 순간 아래에서 위로 슬라이드 + 페이드 인.
/// [index] 가 클수록 진입이 지연되어 list 가 스태거 효과를 갖는다.
class AnimatedEntry extends StatefulWidget {
  final Widget child;
  final int index;
  final Duration baseDuration;
  final Duration perItemDelay;
  final double offsetY;

  const AnimatedEntry({
    super.key,
    required this.child,
    this.index = 0,
    this.baseDuration = kCardEnterDuration,
    this.perItemDelay = const Duration(milliseconds: 60),
    this.offsetY = 24,
  });

  @override
  State<AnimatedEntry> createState() => _AnimatedEntryState();
}

class _AnimatedEntryState extends State<AnimatedEntry>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _opacity;
  late final Animation<Offset> _slide;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.baseDuration,
    );
    _opacity = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _slide = Tween<Offset>(
      begin: Offset(0, widget.offsetY / 100),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));

    final delay = widget.perItemDelay * widget.index;
    Future.delayed(delay, () {
      if (mounted) _controller.forward();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SlideTransition(
      position: _slide,
      child: FadeTransition(
        opacity: _opacity,
        child: widget.child,
      ),
    );
  }
}
