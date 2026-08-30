/// 분석 진행 로더 (전처리 → 분류 → 재질 분석 단계 표시).
library;

import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../../theme/app_theme.dart';

/// 분석 중 로딩 — 썸네일 스캔 오버레이 + 단계 스텝퍼 + 결과 미리 채우기.
/// 단계별 바: 업로드=확정형(실측 바이트 %), AI 분석=불확정형 shimmer.
/// 재질 분석 중 — 시안 16c(개정): 중앙 집중 로딩 — 회전 오라 링 + 상태 문구 + 점 3개.
class AnalysisLoading extends StatefulWidget {
  final int uploadSent;
  final int uploadTotal;
  final bool preprocessDone;
  final bool classifyDone;
  final bool resultDone;
  final bool isSmartCapture;
  final VoidCallback onCancel;

  const AnalysisLoading({
    super.key,
    required this.uploadSent,
    required this.uploadTotal,
    required this.preprocessDone,
    required this.classifyDone,
    required this.resultDone,
    required this.isSmartCapture,
    required this.onCancel,
  });

  @override
  State<AnalysisLoading> createState() => _AnalysisLoadingState();
}

class _AnalysisLoadingState extends State<AnalysisLoading>
    with TickerProviderStateMixin {
  /// 오라 링 회전 (3.2s) — 시안 ggAura.
  late final AnimationController _aura = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 3200),
  )..repeat();

  /// 문구 반짝임·아이콘 회전 (2.4s) — ggTextSweep / ggSpin.
  late final AnimationController _sweep = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2400),
  )..repeat();

  /// 점 3개 펄스 (1.2s) — ggDotPulse.
  late final AnimationController _dots = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1200),
  )..repeat();

  /// 분석 중 순환 문구 — 실제 단계(업로드/분석/마무리)에 맞춰 선택.
  static const _analyzingWords = [
    '재질을 살펴보는 중',
    '표면 질감 확인 중',
    '모양을 맞춰보는 중',
    '색을 확인하는 중',
    '라벨을 찾아보는 중',
  ];
  Timer? _wordTimer;
  int _wordIdx = 0;

  @override
  void initState() {
    super.initState();
    _wordTimer = Timer.periodic(const Duration(milliseconds: 1800), (_) {
      if (mounted) setState(() => _wordIdx++);
    });
  }

  @override
  void dispose() {
    _wordTimer?.cancel();
    _aura.dispose();
    _sweep.dispose();
    _dots.dispose();
    super.dispose();
  }

  bool get _uploadDone =>
      widget.classifyDone ||
      (widget.uploadTotal > 0 && widget.uploadSent >= widget.uploadTotal);

  String get _statusWord {
    if (!_uploadDone) {
      final pct = widget.uploadTotal > 0
          ? (widget.uploadSent / widget.uploadTotal * 100).round()
          : null;
      return pct == null ? '사진 올리는 중' : '사진 올리는 중 $pct%';
    }
    if (widget.classifyDone && !widget.resultDone) return '거의 다 됐어요';
    return _analyzingWords[_wordIdx % _analyzingWords.length];
  }

  @override
  Widget build(BuildContext context) {
    const inner = kInkDeep2;
    return Column(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(28, 0, 28, 40),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // 오라 링 — 2.5px 회전 그라디언트 보더 + 내부 다크 원 + 아이콘
                AnimatedBuilder(
                  animation: _aura,
                  builder: (context, child) => Container(
                    width: 96,
                    height: 96,
                    padding: const EdgeInsets.all(2.5),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: SweepGradient(
                        transform: GradientRotation(_aura.value * 2 * math.pi),
                        colors: [
                          Colors.white.withValues(alpha: 0.08),
                          kAccent400,
                          Colors.white.withValues(alpha: 0.08),
                          kAccent600,
                          Colors.white.withValues(alpha: 0.08),
                        ],
                        stops: const [0.0, 0.28, 0.5, 0.75, 1.0],
                      ),
                    ),
                    child: child,
                  ),
                  child: Container(
                    decoration: const BoxDecoration(
                      color: inner,
                      shape: BoxShape.circle,
                    ),
                    child: const Center(
                      child: CustomPaint(
                        size: Size(38, 38),
                        painter: BlobIconPainter(color: kAccent300),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 26),
                Text(
                  widget.isSmartCapture
                      ? '캡처본 재질을 분석하고 있어요'
                      : '사진 속 재질을 분석하고 있어요',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 23,
                    fontWeight: FontWeight.w600,
                    color: kNeutral100,
                  ),
                ),
                const SizedBox(height: 12),
                // 상태 문구 — 회전 아이콘 + 반짝이는 텍스트 + 펄스 점 3개
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    AnimatedBuilder(
                      animation: _sweep,
                      builder: (context, child) => Transform.rotate(
                        angle: _sweep.value * 2 * math.pi,
                        child: child,
                      ),
                      child: const Icon(
                        Icons.wb_sunny_outlined,
                        size: 14,
                        color: kAccent400,
                      ),
                    ),
                    const SizedBox(width: 7),
                    AnimatedBuilder(
                      animation: _sweep,
                      builder: (context, child) => ShaderMask(
                        blendMode: BlendMode.srcIn,
                        shaderCallback: (rect) => LinearGradient(
                          begin: Alignment(-1 + _sweep.value * 4, 0),
                          end: Alignment(1 + _sweep.value * 4, 0),
                          colors: const [
                            kAccent400,
                            kAccent400,
                            kAccent200,
                            kAccent400,
                            kAccent400,
                          ],
                          stops: const [0.0, 0.38, 0.5, 0.62, 1.0],
                        ).createShader(rect),
                        child: child,
                      ),
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 300),
                        child: Text(
                          _statusWord,
                          key: ValueKey(_statusWord),
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: kAccent400,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 7),
                    AnimatedBuilder(
                      animation: _dots,
                      builder: (context, _) => Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          for (var i = 0; i < 3; i++) ...[
                            if (i > 0) const SizedBox(width: 3),
                            Opacity(
                              opacity: _dotOpacity(_dots.value, i),
                              child: Container(
                                width: 4,
                                height: 4,
                                decoration: const BoxDecoration(
                                  color: kAccent400,
                                  shape: BoxShape.circle,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        Padding(
          padding: EdgeInsets.only(
            bottom: 30 + MediaQuery.viewPaddingOf(context).bottom,
          ),
          child: Material(
            color: Colors.transparent,
            borderRadius: BorderRadius.circular(999),
            child: InkWell(
              borderRadius: BorderRadius.circular(999),
              onTap: widget.onCancel,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 22,
                  vertical: 12,
                ),
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.2),
                  ),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  '분석 취소',
                  style: TextStyle(
                    fontSize: 13.5,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.6),
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  /// ggDotPulse — 0/80/100% 0.25, 40% 1.0, 점마다 0.2s 지연.
  static double _dotOpacity(double t, int i) {
    final local = (t - i * (0.2 / 1.2)) % 1.0;
    final phase = local < 0 ? local + 1 : local;
    if (phase <= 0.4) return 0.25 + 0.75 * (phase / 0.4);
    if (phase <= 0.8) return 1.0 - 0.75 * ((phase - 0.4) / 0.4);
    return 0.25;
  }
}

/// 시안 16c 로딩 아이콘 — 둥근 꽃/톱니 형태(원호 8개) 스트로크.
class BlobIconPainter extends CustomPainter {
  final Color color;
  const BlobIconPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final k = size.width / 24;
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5 * k
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..color = color;
    final r = Radius.circular(3 * k);
    Offset p(double x, double y) => Offset(x * k, y * k);
    final path = Path()
      ..moveTo(12 * k, 3 * k)
      ..arcToPoint(p(9, 6), radius: r, clockwise: false)
      ..arcToPoint(p(6, 9), radius: r, clockwise: false)
      ..arcToPoint(p(6, 15), radius: r, clockwise: false)
      ..arcToPoint(p(9, 18), radius: r, clockwise: false)
      ..arcToPoint(p(15, 18), radius: r, clockwise: false)
      ..arcToPoint(p(18, 15), radius: r, clockwise: false)
      ..arcToPoint(p(18, 9), radius: r, clockwise: false)
      ..arcToPoint(p(15, 6), radius: r, clockwise: false)
      ..arcToPoint(p(12, 3), radius: r, clockwise: false)
      ..close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(BlobIconPainter oldDelegate) => oldDelegate.color != color;
}
