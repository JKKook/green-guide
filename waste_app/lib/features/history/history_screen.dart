import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/di/app_scope.dart';
import '../../core/ui/ds_card.dart';
import '../../data/haptics.dart';
import '../../data/history_repository.dart';
import '../../data/waste_info.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import '../../widgets/criteria_sheet.dart';
import 'widgets/day_picker_sheet.dart';
import 'widgets/day_stack_card.dart';
import 'widgets/full_image_viewer.dart';
import 'widgets/range_sheet.dart';

/// 기록 탭 — 시안 15a: 갤러리형 + 기간(Range) 조회.
/// 날짜 그룹 헤더 아래 2열 썸네일 카드, 상단 기간 칩으로 레인지 캘린더 시트.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}


class _HistoryScreenState extends State<HistoryScreen> {
  final HistoryRepository _repo = AppScope.history;
  List<HistoryEntry> _entries = [];
  bool _loading = true;
  DateTimeRange? _range; // null = 전체 (일 단위로 정규화)

  @override
  void initState() {
    super.initState();
    _load();
    historyRevision.addListener(_load);
  }

  @override
  void dispose() {
    historyRevision.removeListener(_load);
    super.dispose();
  }

  Future<void> _load() async {
    final items = await _repo.recent();
    if (!mounted) return;
    setState(() {
      _entries = items;
      _loading = false;
    });
  }

  static DateTime _dayOf(DateTime t) => DateTime(t.year, t.month, t.day);

  List<HistoryEntry> get _visible {
    final r = _range;
    if (r == null) return _entries;
    return _entries.where((e) {
      final d = _dayOf(e.createdAt);
      return !d.isBefore(r.start) && !d.isAfter(r.end);
    }).toList();
  }

  /// 분류 결과의 요약 태그 — 시안: 재활용 / 비닐·페트 / 일반.
  static (String, bool) _tagOf(HistoryEntry e) {
    final coarse = kFineToCoarse[e.predictedClass] ?? e.predictedClass;
    if (coarse == 'trash' || coarse == 'etc' || coarse == 'non_object') {
      return ('일반', false);
    }
    if (coarse == 'vinyl' || e.predictedClass == 'pet') {
      return ('비닐·페트', true);
    }
    return ('재활용', true);
  }

  String get _rangeChipLabel {
    final r = _range;
    if (r == null) return '전체 기간';
    String f(DateTime d) => '${d.month}. ${d.day}';
    return r.start == r.end ? f(r.start) : '${f(r.start)} – ${f(r.end)}';
  }

  String get _summaryLine {
    final r = _range;
    String f(DateTime d) => '${d.month}월 ${d.day}일';
    final period = r == null
        ? '전체 기간'
        : (r.start == r.end ? f(r.start) : '${f(r.start)} – ${f(r.end)}');
    final items = _visible;
    final general = items.where((e) => !_tagOf(e).$2).length;
    return '$period · 촬영 ${items.length}건 · '
        '재활용 ${items.length - general} · 일반 $general';
  }

  Future<void> _pickRange() async {
    Haptics.selection();
    final picked = await showModalBottomSheet<DateTimeRange?>(
      context: context,
      isScrollControlled: true,
      builder: (_) => RangeSheet(initial: _range),
    );
    if (!mounted || picked == null) return;
    setState(() {
      // 시작·종료가 같은 epoch 0 이면 전체 보기 리셋 신호
      _range = picked.start.millisecondsSinceEpoch == 0 ? null : picked;
    });
  }

  /// 날짜 스택 카드 탭 → 그날의 사진 고르기 시트 → 선택한 기록 뷰어.
  Future<void> _openDayPicker(DateTime day, List<HistoryEntry> items) async {
    Haptics.selection();
    if (items.length == 1) {
      await _openViewer(items.first); // 1건이면 바로 원본
      return;
    }
    final picked = await showModalBottomSheet<HistoryEntry>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => DayPickerSheet(day: day, entries: items, tagOf: _tagOf),
    );
    if (!mounted || picked == null) return;
    await _openViewer(picked);
  }

  /// 상세보기 — 선택 즉시 전체 화면 원본 뷰어 (정보·배출 기준·삭제 포함).
  Future<void> _openViewer(HistoryEntry e) async {
    Haptics.selection();
    final info = infoFor(e.predictedClass) ??
        infoFor(kFineToCoarse[e.predictedClass] ?? '');
    final d = e.createdAt;
    final when = '${d.month}월 ${d.day}일 (${'월화수목금토일'[d.weekday - 1]}) '
        '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}';
    final action = await Navigator.of(context).push<String>(
      PageRouteBuilder(
        opaque: false,
        barrierColor: Colors.black,
        pageBuilder: (_, _, _) => FullImageViewer(
          file: File(e.imagePath),
          title: info?.displayName ?? e.predictedClass,
          subtitle: '$when · 확신 ${(e.confidence * 100).round()}%',
          tag: _tagOf(e),
          canShowCriteria: info != null,
        ),
        transitionsBuilder: (_, anim, _, child) =>
            FadeTransition(opacity: anim, child: child),
      ),
    );
    if (!mounted || action == null) return;
    if (action == 'delete') {
      await _confirmDelete(e);
    } else if (action == 'criteria' && info != null) {
      showCriteriaSheet(context, info);
    }
  }

  Future<void> _confirmDelete(HistoryEntry e) async {
    Haptics.selection();
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('기록 삭제'),
        content: const Text('이 분류 기록을 지울까요?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('취소'),
          ),
          FilledButton.tonal(
            style: FilledButton.styleFrom(
              minimumSize: const Size(0, 44),
              padding: const EdgeInsets.symmetric(horizontal: kSpaceL),
            ),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('삭제'),
          ),
        ],
      ),
    );
    if (ok == true) {
      await _repo.delete(e.id!);
      await _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);

    // 날짜(내림차순) 그룹핑
    final groups = <DateTime, List<HistoryEntry>>{};
    for (final e in _visible) {
      groups.putIfAbsent(_dayOf(e.createdAt), () => []).add(e);
    }
    final days = groups.keys.toList()..sort((a, b) => b.compareTo(a));

    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : RefreshIndicator(
                onRefresh: _load,
                child: ListView(
                  padding: EdgeInsets.fromLTRB(
                    20,
                    kSpaceM,
                    20,
                    kSpaceXXL + 16 + MediaQuery.viewPaddingOf(context).bottom,
                  ),
                  children: [
                    Row(
                      children: [
                        const Text(
                          '기록',
                          style: TextStyle(
                              fontSize: 16, fontWeight: FontWeight.w600),
                        ),
                        const Spacer(),
                        InkWell(
                          borderRadius: BorderRadius.circular(14),
                          onTap: _pickRange,
                          child: DsCard(
                                   tinted: true,
                                   radius: 14,
                                   padding: const EdgeInsets.symmetric(
                                horizontal: 11, vertical: 6),
                                   child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.calendar_month_outlined,
                                    size: 13, color: t.accentChipText),
                                const SizedBox(width: 6),
                                Text(
                                  _rangeChipLabel,
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600,
                                    color: t.accentChipText,
                                  ),
                                ),
                              ],
                            ),
                                 ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _summaryLine,
                      style: TextStyle(fontSize: 12, color: t.muted),
                    ),
                    const SizedBox(height: 16),
                    if (_entries.isEmpty)
                      _EmptyState()
                    else if (days.isEmpty)
                      Padding(
                        padding:
                            const EdgeInsets.symmetric(vertical: kSpaceXXL),
                        child: Column(
                          children: [
                            Icon(Icons.event_busy_outlined,
                                size: 40, color: t.muted),
                            const SizedBox(height: kSpaceS),
                            Text(
                              '이 기간에는 분류 기록이 없어요',
                              style:
                                  TextStyle(fontSize: 13, color: t.muted2),
                            ),
                          ],
                        ),
                      )
                    else
                      for (final day in days) ...[
                        DayStackCard(
                          day: day,
                          entries: groups[day]!,
                          tokens: t,
                          tagOf: _tagOf,
                          onTap: () => _openDayPicker(day, groups[day]!),
                        ),
                        const SizedBox(height: 12),
                      ],
                  ],
                ),
              ),
      ),
    );
  }
}


class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: kSpaceXXL * 2),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(kSpaceL),
            decoration: BoxDecoration(
              color: t.accentChipBg,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.photo_camera_outlined,
                size: 48, color: kAccent400),
          ),
          const SizedBox(height: kSpaceM),
          const Text(
            '아직 분류 기록이 없어요',
            style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: kSpaceXS),
          Text(
            '스마트 촬영으로 분류하면 여기에 기록이 쌓여요',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 12, color: t.muted2),
          ),
        ],
      ),
    );
  }
}
