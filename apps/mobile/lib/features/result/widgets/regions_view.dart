/// 다중재질 영역 빗금 오버레이 + 배지.
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../data/waste_info.dart';

/// 영역 뷰 — 원본 위에 영역별 빗금(서버 렌더)을 깐 오버레이 이미지를 표시하고,
/// 각 재질 영역 중심에 라벨 badge 를 Flutter 로 그림. 오버레이 없으면 원본만.
class RegionsView extends StatelessWidget {
  final File image;
  final PredictionWithRegions? regions;
  const RegionsView({super.key, required this.image, this.regions});

  @override
  Widget build(BuildContext context) {
    final r = regions;
    // 오버레이 없으면 원본 그대로
    if (r == null || !r.hasOverlay) {
      return Image.file(image, fit: BoxFit.cover);
    }

    final overlayBytes = base64Decode(r.overlayBase64!.split(',').last);

    return LayoutBuilder(
      builder: (ctx, constraints) {
        final w = constraints.maxWidth;
        final h = constraints.maxHeight;
        return Stack(
          fit: StackFit.expand,
          children: [
            // 원본 + 영역별 빗금 (서버에서 alpha-blend 렌더한 JPEG)
            Image.memory(overlayBytes, fit: BoxFit.cover),
            // 영역별 재질 라벨 badge — bbox 중심에 배치
            for (final region in r.regions)
              RegionBadge(region: region, areaW: w, areaH: h),
          ],
        );
      },
    );
  }
}

/// 한 재질 영역의 라벨 badge — bbox 중심(상단)에 배치.
class RegionBadge extends StatelessWidget {
  final MaterialRegion region;
  final double areaW;
  final double areaH;
  const RegionBadge({
    super.key,
    required this.region,
    required this.areaW,
    required this.areaH,
  });

  @override
  Widget build(BuildContext context) {
    final info = infoFor(region.slug);
    // 빗금과 같은 색(서버 color_hex) 우선 — 앱 팔레트는 구서버 폴백.
    final accent =
        region.color ?? info?.color ?? Theme.of(context).colorScheme.primary;
    // bbox 중심 x, 상단 y (BoxFit.cover 라 정확 매핑은 어려워 근사 배치).
    const badgeW = 116.0;
    final left = (region.cx * areaW - badgeW / 2).clamp(
      4.0,
      areaW - badgeW - 4,
    );
    final top = (region.bboxNorm[1] * areaH).clamp(6.0, areaH - 36);

    return Positioned(
      left: left,
      top: top,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: accent,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.4),
              blurRadius: 6,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              info?.icon ?? Icons.help_outline,
              color: Colors.white,
              size: 16,
            ),
            const SizedBox(width: 5),
            Text(
              info?.displayName ?? region.slug,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
