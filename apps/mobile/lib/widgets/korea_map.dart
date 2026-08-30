import 'package:flutter/material.dart';

import '../data/korea_map_data.dart';

/// 지도 라벨용 축약명 — 정식 명칭은 폴리곤이 작아 겹침.
const Map<String, String> _kShortNames = {
  '서울특별시': '서울',
  '부산광역시': '부산',
  '대구광역시': '대구',
  '인천광역시': '인천',
  '광주광역시': '광주',
  '대전광역시': '대전',
  '울산광역시': '울산',
  '세종특별자치시': '세종',
  '경기도': '경기',
  '강원특별자치도': '강원',
  '충청북도': '충북',
  '충청남도': '충남',
  '전북특별자치도': '전북',
  '전라남도': '전남',
  '경상북도': '경북',
  '경상남도': '경남',
  '제주특별자치도': '제주',
};

/// 외곽 콜아웃 대상 — 폴리곤이 작아 지도 위 라벨·탭이 어려운 광역시·특별시.
/// 지도 가장자리에 라벨을 빼고 리더선으로 연결하며, 라벨 자체가 탭 타깃.
/// west=왼쪽 열, east=오른쪽 열.
const Map<String, String> _kCalloutSide = {
  '서울특별시': 'west',
  '인천광역시': 'west',
  '세종특별자치시': 'west',
  '대전광역시': 'west',
  '광주광역시': 'west',
  '대구광역시': 'east',
  '울산광역시': 'east',
  '부산광역시': 'east',
};

class _Callout {
  final String name;
  final Rect rect;      // 정규화 라벨 박스
  final Offset anchor;  // 폴리곤 중심 (리더선 끝)
  const _Callout(this.name, this.rect, this.anchor);
}

/// 시도 본토 bbox 중심 (정규화).
final Map<String, Offset> _kCentroids = {
  for (final e in kKoreaProvincePolygons.entries)
    e.key: () {
      final main = e.value.reduce((a, b) => a.length >= b.length ? a : b);
      var minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
      for (final pt in main) {
        if (pt[0] < minX) minX = pt[0];
        if (pt[0] > maxX) maxX = pt[0];
        if (pt[1] < minY) minY = pt[1];
        if (pt[1] > maxY) maxY = pt[1];
      }
      return Offset((minX + maxX) / 2, (minY + maxY) / 2);
    }(),
};

/// 콜아웃 배치 — 앵커 y 순으로 정렬 후 최소 간격을 보장하며 가장자리 열에 적층.
final List<_Callout> _kCallouts = () {
  const w = 0.13, h = 0.052, gap = 0.062;
  final out = <_Callout>[];
  for (final side in ['west', 'east']) {
    final names = _kCalloutSide.entries
        .where((e) => e.value == side)
        .map((e) => e.key)
        .toList()
      ..sort((a, b) => _kCentroids[a]!.dy.compareTo(_kCentroids[b]!.dy));
    var prevBottom = -1.0;
    for (final n in names) {
      var y = _kCentroids[n]!.dy - h / 2;
      if (y < prevBottom + (gap - h)) y = prevBottom + (gap - h);
      final x = side == 'west' ? 0.0 : 1.0 - w;
      out.add(_Callout(n, Rect.fromLTWH(x, y, w, h), _kCentroids[n]!));
      prevBottom = y + h;
    }
  }
  return out;
}();

/// 대한민국 시도 지도 — 지역 선택 Level 1.
///
/// 내장 폴리곤(korea_map_data.dart)을 CustomPainter 로 그리고,
/// 탭 지점을 point-in-polygon 판정해 시도를 선택한다 (외부 지도 SDK 불필요).
class KoreaMap extends StatefulWidget {
  final void Function(String sido) onSelect;
  const KoreaMap({super.key, required this.onSelect});

  @override
  State<KoreaMap> createState() => _KoreaMapState();
}

/// 시도별 대표 면적 (shoelace) — 겹침 판정·그리기 순서용.
/// 경기도 외곽 링이 서울·인천 영역을 포함하므로(구멍 미보존 간략화),
/// 판정은 "포함하는 것 중 최소 면적", 그리기는 "큰 것부터" 여야
/// 서울이 경기 밑에 깔리거나 탭이 경기로 새지 않는다.
final Map<String, double> _kAreas = {
  for (final e in kKoreaProvincePolygons.entries)
    e.key: e.value.fold(0.0, (sum, ring) => sum + _ringArea(ring)),
};

/// 그리기 순서 — 면적 내림차순 (작은 시도가 항상 위에 렌더)
final List<String> _kPaintOrder = kKoreaProvincePolygons.keys.toList()
  ..sort((a, b) => _kAreas[b]!.compareTo(_kAreas[a]!));

double _ringArea(List<List<double>> ring) {
  var a = 0.0;
  for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    a += ring[j][0] * ring[i][1] - ring[i][0] * ring[j][1];
  }
  return a.abs() / 2;
}

class _KoreaMapState extends State<KoreaMap> {
  String? _pressed;   // 탭 피드백 하이라이트

  String? _hitTest(Offset local, Size size) {
    final s = size.shortestSide;
    final dx = (size.width - s) / 2, dy = (size.height - s) / 2;
    final p = Offset((local.dx - dx) / s, (local.dy - dy) / s);

    // 0) 외곽 콜아웃 라벨 — 작은 광역시의 확실한 탭 타깃 (여유 패딩 포함)
    for (final c in _kCallouts) {
      if (c.rect.inflate(0.012).contains(p)) return c.name;
    }

    // 1) 포함하는 시도들 중 최소 면적 선택 (중첩 폴리곤 해소)
    String? best;
    var bestArea = double.infinity;
    for (final entry in kKoreaProvincePolygons.entries) {
      for (final ring in entry.value) {
        if (_inPolygon(p, ring)) {
          final a = _kAreas[entry.key]!;
          if (a < bestArea) {
            best = entry.key;
            bestArea = a;
          }
          break;
        }
      }
    }
    if (best != null) return best;

    // 2) 빗나간 탭 — 최근접 경계점 스냅 (섬·해안 근처 오차 허용)
    var bestD = 0.03 * 0.03;   // 정규화 3% 반경
    for (final entry in kKoreaProvincePolygons.entries) {
      for (final ring in entry.value) {
        for (final pt in ring) {
          final ddx = pt[0] - p.dx, ddy = pt[1] - p.dy;
          final d = ddx * ddx + ddy * ddy;
          if (d < bestD) {
            bestD = d;
            best = entry.key;
          }
        }
      }
    }
    return best;
  }

  /// ray casting point-in-polygon.
  static bool _inPolygon(Offset p, List<List<double>> ring) {
    var inside = false;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      final xi = ring[i][0], yi = ring[i][1];
      final xj = ring[j][0], yj = ring[j][1];
      if ((yi > p.dy) != (yj > p.dy) &&
          p.dx < (xj - xi) * (p.dy - yi) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return AspectRatio(
      aspectRatio: 1,
      child: LayoutBuilder(
        builder: (context, c) {
          final size = Size(c.maxWidth, c.maxHeight);
          return GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTapDown: (d) {
              final hit = _hitTest(d.localPosition, size);
              if (hit != null) setState(() => _pressed = hit);
            },
            onTapUp: (d) {
              final hit = _hitTest(d.localPosition, size);
              setState(() => _pressed = null);
              if (hit != null) widget.onSelect(hit);
            },
            onTapCancel: () => setState(() => _pressed = null),
            child: CustomPaint(
              painter: _KoreaMapPainter(
                fill: cs.primaryContainer.withValues(alpha: 0.45),
                pressedFill: cs.primary.withValues(alpha: 0.55),
                stroke: cs.primary.withValues(alpha: 0.65),
                labelColor: cs.onSurface.withValues(alpha: 0.75),
                labelBg: cs.surfaceContainerHigh,
                pressed: _pressed,
              ),
            ),
          );
        },
      ),
    );
  }
}

class _KoreaMapPainter extends CustomPainter {
  final Color fill;
  final Color pressedFill;
  final Color stroke;
  final Color labelColor;
  final Color labelBg;
  final String? pressed;
  _KoreaMapPainter({
    required this.fill,
    required this.pressedFill,
    required this.stroke,
    required this.labelColor,
    required this.labelBg,
    this.pressed,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final s = size.shortestSide;
    final dx = (size.width - s) / 2, dy = (size.height - s) / 2;
    final strokePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.1
      ..color = stroke;

    for (final name in _kPaintOrder) {
      final rings = kKoreaProvincePolygons[name]!;
      final fillPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = name == pressed ? pressedFill : fill;
      for (final ring in rings) {
        final path = Path()
          ..moveTo(dx + ring[0][0] * s, dy + ring[0][1] * s);
        for (final pt in ring.skip(1)) {
          path.lineTo(dx + pt[0] * s, dy + pt[1] * s);
        }
        path.close();
        canvas.drawPath(path, fillPaint);
        canvas.drawPath(path, strokePaint);
      }
    }

    // 도 단위 라벨 — 폴리곤이 커서 지도 위 표기로 충분 (콜아웃 멤버 제외)
    for (final name in kKoreaProvincePolygons.keys) {
      if (_kCalloutSide.containsKey(name)) continue;
      final c = _kCentroids[name]!;
      final tp = TextPainter(
        text: TextSpan(
          text: _kShortNames[name] ?? name,
          style: TextStyle(
            fontSize: s * 0.030,
            fontWeight: FontWeight.w700,
            color: labelColor,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas,
          Offset(dx + c.dx * s - tp.width / 2, dy + c.dy * s - tp.height / 2));
    }

    // 외곽 콜아웃 — 리더선 + 탭 가능한 라벨 박스
    final leaderPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.8
      ..color = stroke.withValues(alpha: 0.55);
    for (final c in _kCallouts) {
      final isPressed = c.name == pressed;
      final r = Rect.fromLTWH(dx + c.rect.left * s, dy + c.rect.top * s,
          c.rect.width * s, c.rect.height * s);
      final anchor = Offset(dx + c.anchor.dx * s, dy + c.anchor.dy * s);
      final from = Offset(
          c.rect.left < 0.5 ? r.right : r.left, r.center.dy);
      canvas.drawLine(from, anchor, leaderPaint);
      canvas.drawCircle(anchor, 1.8, leaderPaint..style = PaintingStyle.fill);
      leaderPaint.style = PaintingStyle.stroke;

      final rrect = RRect.fromRectAndRadius(r, Radius.circular(r.height / 2));
      canvas.drawRRect(
          rrect,
          Paint()
            ..style = PaintingStyle.fill
            ..color = isPressed ? pressedFill : labelBg);
      canvas.drawRRect(
          rrect,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1
            ..color = stroke);
      final tp = TextPainter(
        text: TextSpan(
          text: _kShortNames[c.name] ?? c.name,
          style: TextStyle(
            fontSize: r.height * 0.52,
            fontWeight: FontWeight.w700,
            color: labelColor,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, r.center - Offset(tp.width / 2, tp.height / 2));
    }
  }

  @override
  bool shouldRepaint(_KoreaMapPainter old) => old.pressed != pressed;
}
