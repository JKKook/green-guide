/// 실시간 카메라 오버레이 — 유리 버튼·가이드 코너·오류 안내.
library;

import 'package:flutter/material.dart';

import '../../../theme/app_theme.dart';

/// 반투명 원형/사각 버튼 (다크 카메라 UI).
class GlassButton extends StatelessWidget {
  final double size;
  final double? radius; // null = 원형
  final bool outlined;
  final bool active;
  final VoidCallback onTap;
  final Widget child;

  /// 아이콘만 있는 버튼이라 스크린리더용 이름이 필요하다.
  final String semanticLabel;
  const GlassButton({
    super.key,
    required this.size,
    this.radius,
    this.outlined = false,
    this.active = false,
    required this.onTap,
    required this.semanticLabel,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    final shape = radius == null
        ? BorderRadius.circular(999)
        : BorderRadius.circular(radius!);
    return Semantics(
      button: true,
      label: semanticLabel,
      child: Material(
        color: Colors.white.withValues(alpha: active ? 0.22 : 0.10),
        borderRadius: shape,
        child: InkWell(
          borderRadius: shape,
          onTap: onTap,
          child: Container(
            width: size,
            height: size,
            decoration: BoxDecoration(
              borderRadius: shape,
              border: outlined
                  ? Border.all(color: Colors.white.withValues(alpha: 0.18))
                  : null,
            ),
            child: Center(child: child),
          ),
        ),
      ),
    );
  }
}

/// 뷰파인더 코너 브래킷 — 34px · 2.5px · 흰색 85%.
class CaptureCorner extends StatelessWidget {
  final bool top;
  final bool left;
  const CaptureCorner({super.key, required this.top, required this.left});

  @override
  Widget build(BuildContext context) {
    final side = BorderSide(
      color: Colors.white.withValues(alpha: 0.85),
      width: 2.5,
    );
    const r = Radius.circular(10);
    return Container(
      width: 34,
      height: 34,
      decoration: BoxDecoration(
        border: Border(
          top: top ? side : BorderSide.none,
          bottom: top ? BorderSide.none : side,
          left: left ? side : BorderSide.none,
          right: left ? BorderSide.none : side,
        ),
        borderRadius: BorderRadius.only(
          topLeft: top && left ? r : Radius.zero,
          topRight: top && !left ? r : Radius.zero,
          bottomLeft: !top && left ? r : Radius.zero,
          bottomRight: !top && !left ? r : Radius.zero,
        ),
      ),
    );
  }
}

class CameraErrorOverlay extends StatelessWidget {
  final String message;
  final bool isPermission;
  final VoidCallback onRetry;
  const CameraErrorOverlay({
    super.key,
    required this.message,
    required this.isPermission,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(kSpaceXL),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isPermission
                  ? Icons.no_photography_outlined
                  : Icons.error_outline,
              color: kNeutral100,
              size: 40,
            ),
            const SizedBox(height: kSpaceS),
            Text(
              message,
              textAlign: TextAlign.center,
              style: const TextStyle(
                color: kNeutral100,
                fontWeight: FontWeight.w700,
              ),
            ),
            if (isPermission) ...[
              const SizedBox(height: kSpaceS),
              const Text(
                '설정 > 앱 > 그린가이드 > 권한 에서\n'
                '카메라를 허용한 뒤 다시 시도해주세요.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
            ],
            const SizedBox(height: kSpaceM),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                OutlinedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: kNeutral100,
                    side: const BorderSide(color: Colors.white54),
                    minimumSize: const Size(0, 44),
                    padding: const EdgeInsets.symmetric(horizontal: kSpaceL),
                  ),
                  child: const Text('돌아가기'),
                ),
                if (isPermission) ...[
                  const SizedBox(width: kSpaceM),
                  FilledButton(
                    onPressed: onRetry,
                    style: FilledButton.styleFrom(
                      minimumSize: const Size(0, 44),
                      padding: const EdgeInsets.symmetric(horizontal: kSpaceL),
                    ),
                    child: const Text('다시 시도'),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}
