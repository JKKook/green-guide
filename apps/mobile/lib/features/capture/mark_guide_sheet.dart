/// 첫 스마트 촬영 안내 — 분리배출 표시가 보이게 찍으면 더 정확하다는 시트.
///
/// 서버는 분리배출 표시(OCR)를 최우선 판정 근거로 쓰므로, 첫 촬영 전에 한 번만 알린다.
library;

import 'package:flutter/material.dart';

import '../../data/settings_store.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';

/// 아직 안 봤으면 안내 시트를 띄우고 본 것으로 기록한다. 닫아도 다시 띄우지 않는다.
Future<void> showMarkGuideOnce(
  BuildContext context, {
  SettingsStore? store,
}) async {
  final s = store ?? SettingsStore();
  if (await s.isMarkGuideShown()) return;
  if (!context.mounted) return;
  await showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    builder: (_) => const MarkGuideSheet(),
  );
  await s.setMarkGuideShown();
}

class MarkGuideSheet extends StatelessWidget {
  const MarkGuideSheet({super.key});

  static const _examples = ['페트', '캔류', '종이', '비닐류'];

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
            const Text(
              '분리배출 표시가 보이게 찍어 주세요',
              style: TextStyle(fontSize: 21, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 6),
            Text(
              '제품에 있는 분리배출 표시가 사진에 함께 나오면 재질을 더 정확하게 알려드려요.',
              style: TextStyle(fontSize: 13.5, height: 1.5, color: t.body),
            ),
            const SizedBox(height: 18),
            Container(
              padding: const EdgeInsets.all(kSpaceL),
              decoration: BoxDecoration(
                color: t.accentChipBg,
                border: Border.all(color: t.accentChipBorder),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  Icon(Icons.recycling, size: 44, color: t.accentChipText),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: [
                        for (final e in _examples)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 10,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: t.surface,
                              border: Border.all(color: t.accentChipBorder),
                              borderRadius: BorderRadius.circular(999),
                            ),
                            child: Text(
                              e,
                              style: TextStyle(
                                fontSize: 12.5,
                                fontWeight: FontWeight.w600,
                                color: t.accentChipText,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            for (final tip in const [
              '병·캔·포장재의 옆면이나 바닥에서 표시를 찾아 함께 담아 주세요',
              '표시가 없어도 괜찮아요. 물건 전체가 보이게 찍으면 돼요',
            ])
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Icon(
                        Icons.check_circle_outline,
                        size: 16,
                        color: t.accentStrong,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        tip,
                        style: TextStyle(fontSize: 13, color: t.muted2),
                      ),
                    ),
                  ],
                ),
              ),
            const SizedBox(height: 14),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('알겠어요, 촬영할게요'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
