/// 온보딩 ④ 주택·빌라 수거 요일 설정.
library;

import 'package:flutter/material.dart';

import '../../../core/di/app_scope.dart';
import '../../../core/ui/ds_card.dart';
import '../../../data/collection_schedule.dart';
import '../../../data/haptics.dart';
import '../../../data/settings_store.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import '../widgets/onboarding_primitives.dart';

class PickupSetupStep extends StatefulWidget {
  final (String, String)? region;
  final bool alarmDefault;
  final Future<void> Function() onDone;
  const PickupSetupStep({super.key, 
    required this.region,
    required this.alarmDefault,
    required this.onDone,
  });

  @override
  State<PickupSetupStep> createState() => _PickupSetupStepState();
}


class _PickupSetupStepState extends State<PickupSetupStep> {
  final SettingsStore _settings = AppScope.settings;
  final ReminderStore _reminders = ReminderStore();

  /// 동네 기본값 — 플라스틱·비닐 수거 요일 (DateTime.weekday 1~7).
  late final Set<int> _days = {
    for (var i = 0; i < 7; i++)
      if (kDefaultWeekSchedule[i] == PickupKind.plasticVinyl) i + 1,
  };
  late bool _alarm = widget.alarmDefault;
  bool _busy = false;

  Future<void> _complete() async {
    setState(() => _busy = true);
    await _settings.setPickupWeekdays(_days.toList()..sort());
    if (_alarm) {
      // 수거일 전날 저녁 8시 알림 — 선택한 요일마다 등록
      final list = await _reminders.load();
      for (final d in _days) {
        list.removeWhere((r) => r.weekday == d);
        list.add(CollectionReminder(
            weekday: d, hour: 20, minute: 0, dayBefore: true));
      }
      await _reminders.save(list);
    }
    await widget.onDone();
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final todayIdx = DateTime.now().weekday - 1;
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
                        '분리 수거 설정',
                        style:
                            TextStyle(fontSize: 26, fontWeight: FontWeight.w600),
                      ),
                    ),
                    StepBadge('3/3'),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  '$place · 주택·빌라 기준으로 알려드려요',
                  style: TextStyle(fontSize: 12.5, color: t.muted2),
                ),
                const SizedBox(height: 18),
                DsCard(
                  radius: 20,
                  padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('우리 집 수거 요일',
                          style: TextStyle(
                              fontSize: 13, fontWeight: FontWeight.w700)),
                      const SizedBox(height: 3),
                      Text('동네 기본값이에요 · 다르면 직접 바꿔주세요',
                          style: TextStyle(fontSize: 11.5, color: t.muted)),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          for (var i = 0; i < 7; i++) ...[
                            if (i > 0) const SizedBox(width: 6),
                            Expanded(
                              child: Builder(builder: (context) {
                                final dayIdx = (todayIdx + i) % 7;
                                final weekday = dayIdx + 1;
                                final on = _days.contains(weekday);
                                return InkWell(
                                  borderRadius: BorderRadius.circular(14),
                                  onTap: () {
                                    Haptics.selection();
                                    setState(() {
                                      if (on) {
                                        _days.remove(weekday);
                                      } else {
                                        _days.add(weekday);
                                      }
                                    });
                                  },
                                  child: Container(
                                    height: 42,
                                    alignment: Alignment.center,
                                    decoration: BoxDecoration(
                                      color: on ? kAccent700 : t.surface,
                                      border: on
                                          ? null
                                          : Border.all(color: t.border),
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    child: Text(
                                      kDayNames[dayIdx],
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: on
                                            ? FontWeight.w700
                                            : FontWeight.w600,
                                        color: on ? kNeutral100 : t.muted,
                                      ),
                                    ),
                                  ),
                                );
                              }),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 14),
                      Container(height: 1, color: t.border),
                      const SizedBox(height: 14),
                      Row(
                        children: [
                          Icon(Icons.schedule, size: 15, color: t.muted),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text('배출 시간대',
                                style: TextStyle(
                                    fontSize: 13,
                                    color: t.dark
                                        ? t.muted2
                                        : const Color(0xFF5D5D60))),
                          ),
                          Text(
                            '일몰 후 ~ 자정',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: t.accentChipText,
                            ),
                          ),
                        ],
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
                            const Text('수거일 전날 알림',
                                style: TextStyle(
                                    fontSize: 14, fontWeight: FontWeight.w700)),
                            const SizedBox(height: 1),
                            Text('저녁 8시에 알려드려요 · 발송은 준비 중',
                                style:
                                    TextStyle(fontSize: 11.5, color: t.muted)),
                          ],
                        ),
                      ),
                      Switch(
                        value: _alarm,
                        activeTrackColor: kAccent700,
                        onChanged: (v) {
                          Haptics.selection();
                          setState(() => _alarm = v);
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(24, 14, 24, 20),
            child: OnboardingButton(
              label: '설정 완료',
              onTap: _busy ? null : _complete,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── ④′ 아파트 — 마무리 (17e) ───────────────────────────────────────────────
