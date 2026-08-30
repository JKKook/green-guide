/// 사진 위 마커 — 탐지된 물건 위치·탭 위치 (cover 역변환 공용).
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../data/waste_info.dart';

/// cover 표시좌표 변환 헬퍼 — 정규화 좌표 → view 픽셀 좌표.
Offset coverToView(Offset norm, Size imgSize, Size view) {
  final scale = math.max(
    view.width / imgSize.width,
    view.height / imgSize.height,
  );
  final dx = (imgSize.width * scale - view.width) / 2;
  final dy = (imgSize.height * scale - view.height) / 2;
  return Offset(
    norm.dx * imgSize.width * scale - dx,
    norm.dy * imgSize.height * scale - dy,
  );
}

/// 객체 후보 번호 마커 — bbox 중심에 번호 뱃지, 탭하면 해당 후보 선택.
class ObjectMarker extends StatelessWidget {
  final int index;
  final ObjectCandidate candidate;
  final Size imgSize;
  final Size view;
  final bool selected;
  final VoidCallback onTap;
  const ObjectMarker({
    super.key,
    required this.index,
    required this.candidate,
    required this.imgSize,
    required this.view,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final p = coverToView(Offset(candidate.cx, candidate.cy), imgSize, view);
    final info = infoForWithRollup(
      candidate.displayClass,
      parentSlug: candidate.coarseClass,
    );
    final color = info?.color ?? Colors.blueGrey;
    const r = 15.0;
    return Positioned(
      left: p.dx - r,
      top: p.dy - r,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: r * 2,
          height: r * 2,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: selected ? color : color.withValues(alpha: 0.85),
            border: Border.all(color: Colors.white, width: selected ? 3 : 1.5),
            boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 5)],
          ),
          alignment: Alignment.center,
          child: Text(
            '${index + 1}',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w800,
              fontSize: 14,
            ),
          ),
        ),
      ),
    );
  }
}

/// 탭 마커 — 사용자가 지목한 지점에 링 표시 (정규화 → cover 표시좌표 변환).
class TapMarker extends StatelessWidget {
  final Offset norm; // 원본 이미지 기준 정규화 좌표
  final Size imgSize;
  final Size view;
  const TapMarker({
    super.key,
    required this.norm,
    required this.imgSize,
    required this.view,
  });

  @override
  Widget build(BuildContext context) {
    final scale = math.max(
      view.width / imgSize.width,
      view.height / imgSize.height,
    );
    final dx = (imgSize.width * scale - view.width) / 2;
    final dy = (imgSize.height * scale - view.height) / 2;
    final px = norm.dx * imgSize.width * scale - dx;
    final py = norm.dy * imgSize.height * scale - dy;
    const r = 18.0;
    return Positioned(
      left: px - r,
      top: py - r,
      child: IgnorePointer(
        child: Container(
          width: r * 2,
          height: r * 2,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 2.5),
            boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 6)],
          ),
          child: const Icon(
            Icons.center_focus_strong,
            color: Colors.white,
            size: 18,
          ),
        ),
      ),
    );
  }
}
