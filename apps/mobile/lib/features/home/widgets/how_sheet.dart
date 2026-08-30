/// 이용 방법 안내 시트.
library;

import 'package:flutter/material.dart';

import '../../../core/ui/ds_card.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';

/// 작동 원리 바텀시트 — 3스텝 + 조건 단계 + 피드백 안내 + 확인.
class HowSheet extends StatelessWidget {
  const HowSheet({super.key});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final t = DsTokens(isDark);
    final steps = [
      (Icons.photo_camera_outlined, '사진 한 장', '촬영·갤러리'),
      (Icons.bolt_outlined, '1차 분류', '기기에서 즉시'),
      (Icons.place_outlined, '동네 기준 안내', '공공데이터 근거'),
    ];

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(kSpaceXL, 0, kSpaceXL, kSpaceXL),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    '스마트 촬영은 이렇게 동작해요',
                    style:
                        TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                ),
                InkWell(
                  borderRadius: BorderRadius.circular(999),
                  onTap: () => Navigator.of(context).pop(),
                  child: Padding(
                    padding: const EdgeInsets.all(kSpaceXS),
                    child: Icon(Icons.close,
                        size: 20,
                        color: t.iconMuted),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 22),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                for (var i = 0; i < steps.length; i++) ...[
                  if (i > 0)
                    Padding(
                      padding: const EdgeInsets.only(top: kSpaceL),
                      child: Icon(Icons.chevron_right,
                          size: 14,
                          color:
                              t.iconMuted),
                    ),
                  Expanded(
                    child: Column(
                      children: [
                        Container(
                          width: 46,
                          height: 46,
                          decoration: const BoxDecoration(
                            color: brandSeed,
                            shape: BoxShape.circle,
                          ),
                          child: Icon(steps[i].$1,
                              size: 22, color: kNeutral100),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          steps[i].$2,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          steps[i].$3,
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 11, color: t.muted2),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 18),
            Row(
              children: [
                Icon(Icons.subdirectory_arrow_right,
                    size: 15,
                    color: t.iconMuted),
                const SizedBox(width: 8),
                DsCard(
                  tinted: true,
                  radius: 999,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.cloud_sync_outlined,
                          size: 14, color: t.accentChipText),
                      const SizedBox(width: 6),
                      Text(
                        '확신이 낮을 때만 · 클라우드 2차 재분류',
                        style: TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w600,
                          color: t.accentChipText,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(height: 1, color: t.border),
            const SizedBox(height: 14),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: t.bannerBg,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(Icons.thumb_up_outlined,
                      size: 16, color: t.accentChipText),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '피드백 한 번이 AI를 더 똑똑하게 만들어요',
                        style: TextStyle(
                            fontSize: 13, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        '결과 화면에서 정확함/수정만 눌러주세요',
                        style: TextStyle(fontSize: 11.5, color: t.muted2),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),
            Material(
              color: kAccent700,
              borderRadius: BorderRadius.circular(kRadiusMedium),
              child: InkWell(
                borderRadius: BorderRadius.circular(kRadiusMedium),
                onTap: () => Navigator.of(context).pop(),
                child: const SizedBox(
                  height: 52,
                  child: Center(
                    child: Text(
                      '확인',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: kNeutral100,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
