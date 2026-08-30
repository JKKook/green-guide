import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/feedback/app_snackbar.dart';
import '../../data/haptics.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import 'gallery_confirm_screen.dart';
import 'live_camera_screen.dart';

/// 스마트 촬영 진입 시트 — 시안 16a: 스마트 촬영 / 갤러리에서 선택.
Future<void> showCaptureEntrySheet(BuildContext context) async {
  Haptics.selection();
  final choice = await showModalBottomSheet<_EntryChoice>(
    context: context,
    showDragHandle: true,
    builder: (_) => const _CaptureEntrySheet(),
  );
  if (choice == null || !context.mounted) return;
  switch (choice) {
    case _EntryChoice.smartCapture:
      await Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => const LiveCameraScreen()),
      );
    case _EntryChoice.gallery:
      await pickFromGalleryAndAnalyze(context);
  }
}

/// 갤러리 선택 → 확인 화면(16d) → 재질 분석.
Future<void> pickFromGalleryAndAnalyze(BuildContext context) async {
  try {
    final picked = await ImagePicker().pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
      maxWidth: 1600,
    );
    if (picked == null || !context.mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => GalleryConfirmScreen(image: File(picked.path)),
      ),
    );
  } on Exception {
    if (!context.mounted) return;
    Haptics.heavy();
    showAppSnackBar(
      context,
      '사진을 불러오지 못했어요. 다시 시도해 주세요.',
      kind: AppSnackKind.error,
    );
  }
}

enum _EntryChoice { smartCapture, gallery }

class _CaptureEntrySheet extends StatelessWidget {
  const _CaptureEntrySheet();

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(kSpaceXL, 0, kSpaceXL, kSpaceXL),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    '어떻게 분석할까요?',
                    style:
                        TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                ),
                Semantics(
                  button: true,
                  label: '닫기',
                  child: InkWell(
                    borderRadius: BorderRadius.circular(999),
                    onTap: () => Navigator.of(context).pop(),
                    child: Padding(
                      padding: const EdgeInsets.all(kSpaceXS),
                      child: Icon(Icons.close, size: 20, color: t.faint),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              '사진 한 장이면 재질과 배출 방법을 알려드려요',
              style: TextStyle(fontSize: 12.5, color: t.muted2),
            ),
            const SizedBox(height: 18),
            // 스마트 촬영 — 주 액션
            Material(
              color: kAccent700,
              borderRadius: BorderRadius.circular(20),
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: () =>
                    Navigator.of(context).pop(_EntryChoice.smartCapture),
                child: Padding(
                  padding: const EdgeInsets.all(kSpaceL),
                  child: Row(
                    children: [
                      Container(
                        width: 46,
                        height: 46,
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.16),
                          borderRadius: BorderRadius.circular(kRadiusMedium),
                        ),
                        child: const Icon(Icons.photo_camera_outlined,
                            size: 23, color: kNeutral100),
                      ),
                      const SizedBox(width: 14),
                      const Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '스마트 촬영',
                              style: TextStyle(
                                fontSize: 15.5,
                                fontWeight: FontWeight.w700,
                                color: kNeutral100,
                              ),
                            ),
                            SizedBox(height: 2),
                            Text(
                              '5초 뒤 자동 캡처 · 흔들림 없이 또렷하게',
                              style: TextStyle(
                                fontSize: 12,
                                color: Color(0xD9F5F5F8),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Icon(Icons.chevron_right,
                          size: 16,
                          color: Colors.white.withValues(alpha: 0.7)),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 10),
            // 갤러리에서 선택
            Material(
              color: t.surface,
              borderRadius: BorderRadius.circular(20),
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: () => Navigator.of(context).pop(_EntryChoice.gallery),
                child: Container(
                  padding: const EdgeInsets.all(kSpaceL),
                  decoration: BoxDecoration(
                    border: Border.all(color: t.border),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 46,
                        height: 46,
                        decoration: BoxDecoration(
                          color: t.accentChipBg,
                          border: Border.all(color: t.accentChipBorder),
                          borderRadius: BorderRadius.circular(kRadiusMedium),
                        ),
                        child: Icon(Icons.image_outlined,
                            size: 22, color: t.accentChipText),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              '갤러리에서 선택',
                              style: TextStyle(
                                  fontSize: 15.5, fontWeight: FontWeight.w700),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              '찍어둔 사진 한 장 골라 분석',
                              style:
                                  TextStyle(fontSize: 12, color: t.muted2),
                            ),
                          ],
                        ),
                      ),
                      Icon(Icons.chevron_right, size: 16, color: t.faint),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.lock_outline, size: 13, color: t.muted),
                const SizedBox(width: 6),
                Text(
                  '사진은 재질 분석에만 사용돼요',
                  style: TextStyle(fontSize: 11.5, color: t.muted),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
