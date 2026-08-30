/// 기록 사진 전체 화면 뷰어.
library;

import 'dart:io';
import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';

/// 전체 화면 원본 뷰어 — 검은 배경 · 핀치 줌/패닝(최대 5배) · 더블탭 확대 · 닫기.
class FullImageViewer extends StatefulWidget {
  final File file;
  final String title;
  final String subtitle;
  final (String, bool)? tag;
  final bool canShowCriteria;
  const FullImageViewer({super.key, 
    required this.file,
    required this.title,
    required this.subtitle,
    this.tag,
    this.canShowCriteria = false,
  });

  @override
  State<FullImageViewer> createState() => _FullImageViewerState();
}


class _FullImageViewerState extends State<FullImageViewer> {
  final TransformationController _zoom = TransformationController();
  bool _chromeVisible = true;

  @override
  void dispose() {
    _zoom.dispose();
    super.dispose();
  }

  void _onDoubleTap(TapDownDetails d) {
    final zoomed = _zoom.value.getMaxScaleOnAxis() > 1.01;
    if (zoomed) {
      _zoom.value = Matrix4.identity();
      return;
    }
    // 더블탭 지점을 중심으로 2.5배
    final p = d.localPosition;
    _zoom.value = Matrix4.identity()
      ..translateByDouble(-p.dx * 1.5, -p.dy * 1.5, 0, 1)
      ..scaleByDouble(2.5, 2.5, 1, 1);
  }

  @override
  Widget build(BuildContext context) {
    TapDownDetails? lastTap;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          GestureDetector(
            onTap: () => setState(() => _chromeVisible = !_chromeVisible),
            onDoubleTapDown: (d) => lastTap = d,
            onDoubleTap: () {
              if (lastTap != null) _onDoubleTap(lastTap!);
            },
            child: InteractiveViewer(
              transformationController: _zoom,
              minScale: 1,
              maxScale: 5,
              child: Center(
                child: Image.file(widget.file, fit: BoxFit.contain),
              ),
            ),
          ),
          // 상단 크롬 — 닫기 + 제목
          AnimatedOpacity(
            opacity: _chromeVisible ? 1 : 0,
            duration: const Duration(milliseconds: 180),
            child: IgnorePointer(
              ignoring: !_chromeVisible,
              child: Align(
                alignment: Alignment.topCenter,
                child: Container(
                  padding: EdgeInsets.fromLTRB(
                      16, MediaQuery.viewPaddingOf(context).top + 8, 16, 14),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Colors.black.withValues(alpha: 0.6),
                        Colors.transparent,
                      ],
                    ),
                  ),
                  child: Row(
                    children: [
                      Material(
                        color: Colors.white.withValues(alpha: 0.12),
                        shape: const CircleBorder(),
                        child: InkWell(
                          customBorder: const CircleBorder(),
                          onTap: () => Navigator.of(context).pop(),
                          child: const SizedBox(
                            width: 38,
                            height: 38,
                            child: Icon(Icons.close,
                                size: 18, color: kNeutral100),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(widget.title,
                                style: const TextStyle(
                                    fontSize: 15,
                                    fontWeight: FontWeight.w700,
                                    color: kNeutral100)),
                            Text(widget.subtitle,
                                style: TextStyle(
                                    fontSize: 11.5,
                                    color: Colors.white.withValues(alpha: 0.7))),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          // 하단 — 태그 + 액션(배출 기준 보기 · 삭제) + 힌트
          AnimatedOpacity(
            opacity: _chromeVisible ? 1 : 0,
            duration: const Duration(milliseconds: 180),
            child: IgnorePointer(
              ignoring: !_chromeVisible,
              child: Align(
                alignment: Alignment.bottomCenter,
                child: Container(
                  padding: EdgeInsets.fromLTRB(20, 24, 20,
                      MediaQuery.viewPaddingOf(context).bottom + 16),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.bottomCenter,
                      end: Alignment.topCenter,
                      colors: [
                        Colors.black.withValues(alpha: 0.75),
                        Colors.transparent,
                      ],
                    ),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (widget.tag != null)
                        Align(
                          alignment: Alignment.centerLeft,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 9, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.white.withValues(alpha: 0.14),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Text(
                              widget.tag!.$1,
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: widget.tag!.$2 ? kAccent200 : kNeutral100,
                              ),
                            ),
                          ),
                        ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: Material(
                              color: Colors.white.withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(14),
                              child: InkWell(
                                borderRadius: BorderRadius.circular(14),
                                onTap: () => Navigator.of(context).pop('delete'),
                                child: const SizedBox(
                                  height: 48,
                                  child: Center(
                                    child: Text('삭제',
                                        style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.w600,
                                            color: kNeutral100)),
                                  ),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            flex: 2,
                            child: Material(
                              color: widget.canShowCriteria
                                  ? kAccent700
                                  : Colors.white.withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(14),
                              child: InkWell(
                                borderRadius: BorderRadius.circular(14),
                                onTap: widget.canShowCriteria
                                    ? () => Navigator.of(context).pop('criteria')
                                    : null,
                                child: const SizedBox(
                                  height: 48,
                                  child: Center(
                                    child: Text('배출 기준 보기',
                                        style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.w700,
                                            color: kNeutral100)),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Center(
                        child: Text(
                          '두 손가락으로 확대 · 더블탭 확대/원래대로 · 탭하면 정보 숨김',
                          style: TextStyle(
                              fontSize: 11,
                              color: Colors.white.withValues(alpha: 0.55)),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
