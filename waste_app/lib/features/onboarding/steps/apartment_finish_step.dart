/// 온보딩 마무리 — 아파트·오피스텔.
library;

import 'package:flutter/material.dart';

import '../../../core/di/app_scope.dart';
import '../../../core/ui/ds_card.dart';
import '../../../data/haptics.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import '../widgets/onboarding_primitives.dart';

class ApartmentFinishStep extends StatefulWidget {
  final (String, String)? region;
  final Future<void> Function() onDone;
  const ApartmentFinishStep({super.key, required this.region, required this.onDone});

  @override
  State<ApartmentFinishStep> createState() => _ApartmentFinishStepState();
}


class _ApartmentFinishStepState extends State<ApartmentFinishStep> {
  bool _tips = false;
  bool _busy = false;

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final place = widget.region?.$2 ?? '우리 동네';
    return SafeArea(
      child: Column(
        children: [
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
              children: [
                const Row(
                  children: [
                    Expanded(
                      child: Text(
                        '거의 다 됐어요',
                        style:
                            TextStyle(fontSize: 26, fontWeight: FontWeight.w600),
                      ),
                    ),
                    StepBadge('3/3'),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  '$place · 아파트 기준으로 알려드려요',
                  style: TextStyle(fontSize: 12.5, color: t.muted2),
                ),
                const SizedBox(height: 18),
                DsCard(
                  tinted: true,
                  radius: 20,
                  padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        width: 38,
                        height: 38,
                        decoration: BoxDecoration(
                          color: t.bannerBg,
                          borderRadius: BorderRadius.circular(13),
                        ),
                        child: Icon(Icons.apartment_outlined,
                            size: 19, color: t.accentChipText),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '아파트는 수거 요일 설정이 필요 없어요',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: t.accentDeep,
                              ),
                            ),
                            const SizedBox(height: 3),
                            Text(
                              '단지 내 분리배출장에 상시 배출할 수 있어 수거 요일·배출 시간대 설정 화면을 건너뛰어요',
                              style: TextStyle(
                                fontSize: 12.5,
                                height: 1.5,
                                color: t.accentChipText,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                DsCard(
                  radius: 20,
                  padding: const EdgeInsets.fromLTRB(20, 16, 14, 16),
                  child: Row(
                    children: [
                      Icon(Icons.notifications_none,
                          size: 19, color: t.accentChipText),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('분리배출 꿀팁 알림',
                                style: TextStyle(
                                    fontSize: 14, fontWeight: FontWeight.w700)),
                            const SizedBox(height: 1),
                            Text('헷갈리는 품목 배출법을 가끔 알려드려요 · 발송은 준비 중',
                                style:
                                    TextStyle(fontSize: 11.5, color: t.muted)),
                          ],
                        ),
                      ),
                      Switch(
                        value: _tips,
                        activeTrackColor: kAccent700,
                        onChanged: (v) {
                          Haptics.selection();
                          setState(() => _tips = v);
                        },
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
                Center(
                  child: Text(
                    '주거 형태는 설정에서 언제든 변경할 수 있어요',
                    style: TextStyle(fontSize: 11.5, color: t.muted),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(24, 14, 24, 20),
            child: OnboardingButton(
              label: '그린가이드 시작하기',
              onTap: _busy
                  ? null
                  : () async {
                      setState(() => _busy = true);
                      await AppScope.settings.setTipsNotificationEnabled(_tips);
                      await widget.onDone();
                    },
            ),
          ),
        ],
      ),
    );
  }
}
