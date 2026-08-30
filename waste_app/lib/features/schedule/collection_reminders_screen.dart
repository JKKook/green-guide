/// 수거일 알림 목록 화면.
library;

import 'package:flutter/material.dart';
import '../../core/ui/ds_card.dart';
import '../../data/collection_schedule.dart';
import '../../data/haptics.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import 'reminder_sheet.dart';

/// 수거일 알림 관리 — 시안 4c: 등록된 알림 목록.
class CollectionRemindersScreen extends StatefulWidget {
  const CollectionRemindersScreen({super.key});

  @override
  State<CollectionRemindersScreen> createState() =>
      _CollectionRemindersScreenState();
}


class _CollectionRemindersScreenState extends State<CollectionRemindersScreen> {
  final ReminderStore _store = ReminderStore();
  List<CollectionReminder> _reminders = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final list = await _store.load();
    list.sort((a, b) => a.weekday.compareTo(b.weekday));
    if (!mounted) return;
    setState(() {
      _reminders = list;
      _loading = false;
    });
  }

  Future<void> _add() async {
    Haptics.selection();
    final result = await showReminderSheet(
      context,
      weekday: DateTime.now().weekday,
      allowWeekdayPick: true,
    );
    if (result == null || result.action != ReminderAction.save) return;
    final list = await _store.load();
    list.removeWhere((r) => r.weekday == result.reminder!.weekday);
    list.add(result.reminder!);
    await _store.save(list);
    await _load();
  }

  Future<void> _edit(CollectionReminder reminder) async {
    Haptics.selection();
    final result = await showReminderSheet(
      context,
      existing: reminder,
      weekday: reminder.weekday,
    );
    if (result == null) return;
    final list = await _store.load();
    list.removeWhere((r) => r.weekday == reminder.weekday);
    if (result.action == ReminderAction.save) {
      list.add(result.reminder!);
    } else {
      list.add(reminder.copyWith(enabled: false));
    }
    await _store.save(list);
    await _load();
  }

  Future<void> _toggle(CollectionReminder reminder, bool enabled) async {
    Haptics.selection();
    final list = await _store.load();
    final i = list.indexWhere((r) => r.weekday == reminder.weekday);
    if (i >= 0) list[i] = list[i].copyWith(enabled: enabled);
    await _store.save(list);
    await _load();
  }

  Future<void> _delete(CollectionReminder reminder) async {
    final list = await _store.load();
    list.removeWhere((r) => r.weekday == reminder.weekday);
    await _store.save(list);
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 0,
        title: const Text(
          '수거일 알림',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        actions: [
          InkWell(
            borderRadius: BorderRadius.circular(999),
            onTap: _add,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
              decoration: BoxDecoration(
                color: t.surface,
                border: Border.all(
                  color: t.accentSoft,
                ),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.add, size: 12, color: t.accentDeep),
                  const SizedBox(width: 5),
                  Text(
                    '알림 추가',
                    style: TextStyle(
                      fontSize: 11.5,
                      fontWeight: FontWeight.w700,
                      color: t.accentDeep,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(width: kSpaceL),
        ],
      ),
      body: SafeArea(
        top: false,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : ListView(
                padding: const EdgeInsets.fromLTRB(20, kSpaceS, 20, kSpaceXL),
                children: [
                  Text(
                    _reminders.isEmpty
                        ? '등록된 알림이 없어요 · 수거일마다 반복돼요'
                        : '등록된 알림 ${_reminders.length}개 · 수거일마다 반복돼요',
                    style: TextStyle(fontSize: 11, color: t.muted2),
                  ),
                  const SizedBox(height: 6),
                  // OS 알림 연동 전 — 설정만 저장된다는 점을 분명히 안내.
                  Row(
                    children: [
                      Icon(Icons.info_outline, size: 12, color: t.muted2),
                      const SizedBox(width: 5),
                      Expanded(
                        child: Text(
                          kReminderPendingNote,
                          style: TextStyle(fontSize: 11, color: t.muted2),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  for (final r in _reminders)
                    Dismissible(
                      key: ValueKey('reminder-${r.weekday}'),
                      direction: DismissDirection.endToStart,
                      onDismissed: (_) => _delete(r),
                      background: Container(
                        alignment: Alignment.centerRight,
                        padding: const EdgeInsets.only(right: 20),
                        margin: const EdgeInsets.only(bottom: kSpaceS),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.errorContainer,
                          borderRadius: BorderRadius.circular(kRadiusMedium),
                        ),
                        child: Icon(
                          Icons.delete_outline,
                          color:
                              Theme.of(context).colorScheme.onErrorContainer,
                        ),
                      ),
                      child: Opacity(
                        opacity: r.enabled ? 1 : 0.62,
                        child: DsCard(
                                 elevated: true,
                                 margin: const EdgeInsets.only(bottom: kSpaceS),
                                 child: InkWell(
                            borderRadius: BorderRadius.circular(kRadiusMedium),
                            onTap: () => _edit(r),
                            child: Padding(
                              padding:
                                  const EdgeInsets.fromLTRB(15, 14, 15, 14),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          r.timeLabel,
                                          style: const TextStyle(
                                            fontSize: 24,
                                            fontWeight: FontWeight.w600,
                                            height: 1,
                                          ),
                                        ),
                                        const SizedBox(height: 3),
                                        Text(
                                          '${kDayNames[r.weekday - 1]}요일 · '
                                          '${r.pickup.fullLabel} · '
                                          '${r.dayBefore ? '하루 전' : '당일'}',
                                          style: TextStyle(
                                            fontSize: 11,
                                            color: t.muted2,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Switch(
                                    value: r.enabled,
                                    activeTrackColor: brandSeed,
                                    onChanged: (v) => _toggle(r, v),
                                  ),
                                ],
                              ),
                            ),
                          ),
                               ),
                      ),
                    ),
                  const SizedBox(height: 8),
                  DsCard(
                    padding: const EdgeInsets.fromLTRB(15, 13, 15, 13),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(top: 1),
                          child:
                              Icon(Icons.info_outline, size: 15, color: t.muted),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            '알림을 누르면 시간·요일을 수정할 수 있어요. '
                            '왼쪽으로 밀면 삭제돼요.',
                            style: TextStyle(
                              fontSize: 11,
                              height: 1.5,
                              color: t.muted2,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
