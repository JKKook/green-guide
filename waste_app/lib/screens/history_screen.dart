import 'dart:io';
import 'package:flutter/material.dart';

import '../data/haptics.dart';
import '../data/history_repository.dart';
import '../data/waste_info.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/criteria_sheet.dart';

/// 기록 탭 — 시안 15a: 갤러리형 + 기간(Range) 조회.
/// 날짜 그룹 헤더 아래 2열 썸네일 카드, 상단 기간 칩으로 레인지 캘린더 시트.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final HistoryRepository _repo = HistoryRepository();
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
      builder: (_) => _RangeSheet(initial: _range),
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
      builder: (_) => _DayPickerSheet(day: day, entries: items, tagOf: _tagOf),
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
        pageBuilder: (_, _, _) => _FullImageViewer(
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
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 11, vertical: 6),
                            decoration: BoxDecoration(
                              color: t.accentChipBg,
                              border: Border.all(
                                color: t.dark ? kAccent700 : kAccent300,
                              ),
                              borderRadius: BorderRadius.circular(14),
                            ),
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
                        _DayStackCard(
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

/// 날짜 스택 카드 — 그날의 사진을 겹쳐 쌓은 더미 + 요약. 탭하면 사진 고르기.
class _DayStackCard extends StatelessWidget {
  final DateTime day;
  final List<HistoryEntry> entries;
  final DsTokens tokens;
  final (String, bool) Function(HistoryEntry) tagOf;
  final VoidCallback onTap;
  const _DayStackCard({
    required this.day,
    required this.entries,
    required this.tokens,
    required this.tagOf,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final t = tokens;
    final general = entries.where((e) => !tagOf(e).$2).length;
    final names = <String>{
      for (final e in entries)
        (infoFor(e.predictedClass) ??
                infoFor(kFineToCoarse[e.predictedClass] ?? ''))
            ?.displayName ??
            e.predictedClass,
    }.toList();
    final shown = names.take(3).toList();
    final more = names.length - shown.length;
    final stack = entries.take(3).toList();

    return Material(
      color: t.surface,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.fromLTRB(14, 14, 12, 14),
          decoration: BoxDecoration(
            border: Border.all(color: t.border),
            borderRadius: BorderRadius.circular(18),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF2B2B2D).withValues(alpha: 0.14),
                offset: const Offset(0, 1),
                blurRadius: 2,
              ),
            ],
          ),
          child: Row(
            children: [
              // 겹쳐진 썸네일 더미 (최대 3장, 뒤로 갈수록 살짝 기울고 흐려짐)
              SizedBox(
                width: 64.0 + 14 * (stack.length - 1),
                height: 72,
                child: Stack(
                  children: [
                    for (var i = stack.length - 1; i >= 0; i--)
                      Positioned(
                        left: 14.0 * i,
                        top: i == 0 ? 4 : (i == 1 ? 0 : 2),
                        child: Transform.rotate(
                          angle: i == 0 ? 0 : (i == 1 ? 0.09 : -0.07),
                          child: Opacity(
                            opacity: i == 0 ? 1 : (i == 1 ? 0.85 : 0.7),
                            child: _Thumb(
                                entry: stack[i],
                                tokens: t,
                                size: 64,
                                recyclable: tagOf(stack[i]).$2),
                          ),
                        ),
                      ),
                    if (entries.length > 3)
                      Positioned(
                        right: 0,
                        bottom: 0,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 7, vertical: 3),
                          decoration: BoxDecoration(
                            color: kAccent700,
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            '+${entries.length - 3}',
                            style: const TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w700,
                              color: kNeutral100,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${day.month}월 ${day.day}일 '
                      '(${'월화수목금토일'[day.weekday - 1]})',
                      style: const TextStyle(
                          fontSize: 14.5, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '${entries.length}건 · 재활용 ${entries.length - general} · 일반 $general',
                      style: TextStyle(fontSize: 11.5, color: t.muted),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 5,
                      runSpacing: 5,
                      children: [
                        for (final n in shown)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: t.accentChipBg,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              n,
                              style: TextStyle(
                                fontSize: 10.5,
                                fontWeight: FontWeight.w600,
                                color: t.accentChipText,
                              ),
                            ),
                          ),
                        if (more > 0)
                          Padding(
                            padding: const EdgeInsets.only(top: 3),
                            child: Text(
                              '외 $more',
                              style: TextStyle(fontSize: 10.5, color: t.muted),
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              Icon(Icons.chevron_right, size: 18, color: t.faint),
            ],
          ),
        ),
      ),
    );
  }
}

/// 썸네일 — 재질 아이콘 타일 (사진은 원본 뷰어에서만 렌더링).
class _Thumb extends StatelessWidget {
  final HistoryEntry entry;
  final DsTokens tokens;
  final double size;
  final bool recyclable;
  const _Thumb({
    required this.entry,
    required this.tokens,
    required this.size,
    this.recyclable = true,
  });

  @override
  Widget build(BuildContext context) {
    final info = infoFor(entry.predictedClass) ??
        infoFor(kFineToCoarse[entry.predictedClass] ?? '');
    final iconSize = size.isFinite ? size * 0.4 : 28.0;
    return Container(
      width: size,
      height: size,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: recyclable ? tokens.accentChipBg : tokens.border,
        border: Border.all(color: tokens.surface, width: 2),
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF2B2B2D).withValues(alpha: 0.18),
            offset: const Offset(0, 2),
            blurRadius: 6,
          ),
        ],
      ),
      child: Center(
        child: Icon(
          info?.icon ?? Icons.help_outline,
          size: iconSize,
          color: recyclable ? tokens.accentStrong : tokens.muted2,
        ),
      ),
    );
  }
}

/// 그날의 사진 고르기 시트 — 3열 썸네일 그리드, 탭하면 해당 기록 반환.
class _DayPickerSheet extends StatelessWidget {
  final DateTime day;
  final List<HistoryEntry> entries;
  final (String, bool) Function(HistoryEntry) tagOf;
  const _DayPickerSheet({
    required this.day,
    required this.entries,
    required this.tagOf,
  });

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final maxH = MediaQuery.sizeOf(context).height * 0.7;
    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxHeight: maxH),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${day.month}월 ${day.day}일 (${'월화수목금토일'[day.weekday - 1]}) · ${entries.length}건',
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 4),
              Text(
                '보고 싶은 사진을 골라주세요',
                style: TextStyle(fontSize: 12.5, color: t.muted2),
              ),
              const SizedBox(height: 14),
              Flexible(
                child: GridView.builder(
                  shrinkWrap: true,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3,
                    mainAxisSpacing: 10,
                    crossAxisSpacing: 10,
                    childAspectRatio: 0.78,
                  ),
                  itemCount: entries.length,
                  itemBuilder: (context, i) {
                    final e = entries[i];
                    final info = infoFor(e.predictedClass) ??
                        infoFor(kFineToCoarse[e.predictedClass] ?? '');
                    final (label, recyclable) = tagOf(e);
                    final time =
                        '${e.createdAt.hour.toString().padLeft(2, '0')}:${e.createdAt.minute.toString().padLeft(2, '0')}';
                    return InkWell(
                      borderRadius: BorderRadius.circular(14),
                      onTap: () {
                        Haptics.selection();
                        Navigator.of(context).pop(e);
                      },
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          AspectRatio(
                            aspectRatio: 1,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                _Thumb(
                                    entry: e,
                                    tokens: t,
                                    size: double.infinity,
                                    recyclable: recyclable),
                                Positioned(
                                  left: 6,
                                  bottom: 6,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 6, vertical: 2),
                                    decoration: BoxDecoration(
                                      color: const Color(0xFF131518)
                                          .withValues(alpha: 0.6),
                                      borderRadius: BorderRadius.circular(999),
                                    ),
                                    child: Text(
                                      label,
                                      style: TextStyle(
                                        fontSize: 9.5,
                                        fontWeight: FontWeight.w700,
                                        color: recyclable
                                            ? kAccent200
                                            : kNeutral100,
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            info?.displayName ?? e.predictedClass,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                                fontSize: 12, fontWeight: FontWeight.w700),
                          ),
                          Text(
                            time,
                            style: TextStyle(fontSize: 10.5, color: t.muted),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 기간 선택 바텀시트 — 시안 15b: 시작/종료 + 프리셋 + 레인지 캘린더.
class _RangeSheet extends StatefulWidget {
  final DateTimeRange? initial;
  const _RangeSheet({this.initial});

  @override
  State<_RangeSheet> createState() => _RangeSheetState();
}

class _RangeSheetState extends State<_RangeSheet> {
  late DateTime _month;   // 표시 중인 달 (1일)
  DateTime? _start;
  DateTime? _end;
  bool _pickingEnd = false;
  String? _preset;

  static DateTime _dayOf(DateTime t) => DateTime(t.year, t.month, t.day);

  @override
  void initState() {
    super.initState();
    _start = widget.initial?.start;
    _end = widget.initial?.end;
    final base = _end ?? DateTime.now();
    _month = DateTime(base.year, base.month);
  }

  void _applyPreset(String label) {
    Haptics.selection();
    final today = _dayOf(DateTime.now());
    final (start, end) = switch (label) {
      '오늘' => (today, today),
      '최근 7일' => (today.subtract(const Duration(days: 6)), today),
      '최근 30일' => (today.subtract(const Duration(days: 29)), today),
      _ => (DateTime(today.year, today.month), today), // 이번 달
    };
    setState(() {
      _preset = label;
      _start = start;
      _end = end;
      _pickingEnd = false;
      _month = DateTime(end.year, end.month);
    });
  }

  void _onDayTap(DateTime day) {
    Haptics.selection();
    setState(() {
      _preset = null;
      if (!_pickingEnd || _start == null) {
        _start = day;
        _end = null;
        _pickingEnd = true;
      } else {
        if (day.isBefore(_start!)) {
          _end = _start;
          _start = day;
        } else {
          _end = day;
        }
        _pickingEnd = false;
      }
    });
  }

  String _fmt(DateTime? d) => d == null
      ? '선택'
      : '${d.month}월 ${d.day}일 (${'월화수목금토일'[d.weekday - 1]})';

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final today = _dayOf(DateTime.now());
    final firstWeekday = _month.weekday % 7; // 일요일 시작 캘린더
    final daysInMonth = DateTime(_month.year, _month.month + 1, 0).day;

    bool inRange(DateTime d) {
      if (_start == null || _end == null) return false;
      return !d.isBefore(_start!) && !d.isAfter(_end!);
    }

    bool isEdge(DateTime d) =>
        (_start != null && d == _start) || (_end != null && d == _end);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(22, 8, 22, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: 18),
                decoration: BoxDecoration(
                  color: t.dark ? const Color(0xFF5D5D60) : kNeutral300,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            Row(
              children: [
                const Text(
                  '기간 선택',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
                ),
                const Spacer(),
                InkWell(
                  borderRadius: BorderRadius.circular(999),
                  onTap: () => Navigator.of(context).pop(),
                  child: Padding(
                    padding: const EdgeInsets.all(4),
                    child: Icon(Icons.close, size: 20, color: t.muted),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: InkWell(
                    borderRadius: BorderRadius.circular(14),
                    onTap: () => setState(() => _pickingEnd = false),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 9),
                      decoration: BoxDecoration(
                        color: t.surface,
                        border: Border.all(
                          color: !_pickingEnd ? kAccent500 : t.border,
                          width: !_pickingEnd ? 1.5 : 1,
                        ),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '시작',
                            style: TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w700,
                              color: t.muted,
                            ),
                          ),
                          const SizedBox(height: 1),
                          Text(
                            _fmt(_start),
                            style: const TextStyle(
                                fontSize: 14, fontWeight: FontWeight.w700),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  child:
                      Icon(Icons.arrow_forward, size: 15, color: t.faint),
                ),
                Expanded(
                  child: InkWell(
                    borderRadius: BorderRadius.circular(14),
                    onTap: () {
                      if (_start != null) {
                        setState(() => _pickingEnd = true);
                      }
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 9),
                      decoration: BoxDecoration(
                        color: t.surface,
                        border: Border.all(
                          color: _pickingEnd ? kAccent500 : t.border,
                          width: _pickingEnd ? 1.5 : 1,
                        ),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '종료',
                            style: TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w700,
                              color: t.muted,
                            ),
                          ),
                          const SizedBox(height: 1),
                          Text(
                            _fmt(_end),
                            style: const TextStyle(
                                fontSize: 14, fontWeight: FontWeight.w700),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                for (final p in const ['오늘', '최근 7일', '최근 30일', '이번 달'])
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(12),
                      onTap: () => _applyPreset(p),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 11, vertical: 6),
                        decoration: BoxDecoration(
                          color: _preset == p ? kAccent700 : t.surface,
                          border: _preset == p
                              ? null
                              : Border.all(color: t.border),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          p,
                          style: TextStyle(
                            fontSize: 11.5,
                            fontWeight: _preset == p
                                ? FontWeight.w700
                                : FontWeight.w600,
                            color: _preset == p ? kNeutral100 : t.muted2,
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                InkWell(
                  borderRadius: BorderRadius.circular(999),
                  onTap: () => setState(() =>
                      _month = DateTime(_month.year, _month.month - 1)),
                  child: Padding(
                    padding: const EdgeInsets.all(4),
                    child:
                        Icon(Icons.chevron_left, size: 18, color: t.muted),
                  ),
                ),
                Expanded(
                  child: Text(
                    '${_month.year}년 ${_month.month}월',
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                        fontSize: 14, fontWeight: FontWeight.w700),
                  ),
                ),
                InkWell(
                  borderRadius: BorderRadius.circular(999),
                  onTap: () => setState(() =>
                      _month = DateTime(_month.year, _month.month + 1)),
                  child: Padding(
                    padding: const EdgeInsets.all(4),
                    child:
                        Icon(Icons.chevron_right, size: 18, color: t.muted),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            GridView.count(
              crossAxisCount: 7,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              mainAxisSpacing: 3,
              crossAxisSpacing: 3,
              childAspectRatio: 1.35,
              children: [
                for (final d in const ['일', '월', '화', '수', '목', '금', '토'])
                  Center(
                    child: Text(
                      d,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        color: t.muted,
                      ),
                    ),
                  ),
                for (var i = 0; i < firstWeekday; i++) const SizedBox(),
                for (var day = 1; day <= daysInMonth; day++)
                  Builder(builder: (context) {
                    final date =
                        DateTime(_month.year, _month.month, day);
                    final edge = isEdge(date);
                    final ranged = inRange(date);
                    final isToday = date == today;
                    final future = date.isAfter(today);
                    return InkWell(
                      borderRadius: BorderRadius.circular(999),
                      onTap: future ? null : () => _onDayTap(date),
                      child: Container(
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: edge
                              ? kAccent700
                              : ranged
                                  ? t.accentChipBg
                                  : null,
                          shape: edge ? BoxShape.circle : BoxShape.rectangle,
                          border: !edge && !ranged && isToday
                              ? Border.all(color: kAccent400)
                              : null,
                          borderRadius: edge
                              ? null
                              : (isToday && !ranged
                                  ? BorderRadius.circular(999)
                                  : null),
                        ),
                        child: Text(
                          '$day',
                          style: TextStyle(
                            fontSize: 12.5,
                            fontWeight: edge
                                ? FontWeight.w700
                                : ranged
                                    ? FontWeight.w600
                                    : FontWeight.w400,
                            color: edge
                                ? kNeutral100
                                : ranged
                                    ? t.accentDeep
                                    : future
                                        ? t.faint
                                        : t.muted2,
                          ),
                        ),
                      ),
                    );
                  }),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: InkWell(
                    borderRadius: BorderRadius.circular(16),
                    onTap: () => Navigator.of(context).pop(
                      DateTimeRange(
                        start: DateTime.fromMillisecondsSinceEpoch(0),
                        end: DateTime.fromMillisecondsSinceEpoch(0),
                      ),
                    ),
                    child: Container(
                      height: 52,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        border: Border.all(
                          color:
                              t.dark ? const Color(0xFF5D5D60) : kNeutral300,
                        ),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        '전체 보기',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: t.muted2,
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  flex: 2,
                  child: Material(
                    color: _start != null && _end != null
                        ? kAccent700
                        : t.border,
                    borderRadius: BorderRadius.circular(16),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(16),
                      onTap: _start != null && _end != null
                          ? () => Navigator.of(context).pop(
                                DateTimeRange(start: _start!, end: _end!),
                              )
                          : null,
                      child: SizedBox(
                        height: 52,
                        child: Center(
                          child: Text(
                            '이 기간으로 조회',
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                              color: _start != null && _end != null
                                  ? kNeutral100
                                  : t.muted,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
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

/// 전체 화면 원본 뷰어 — 검은 배경 · 핀치 줌/패닝(최대 5배) · 더블탭 확대 · 닫기.
class _FullImageViewer extends StatefulWidget {
  final File file;
  final String title;
  final String subtitle;
  final (String, bool)? tag;
  final bool canShowCriteria;
  const _FullImageViewer({
    required this.file,
    required this.title,
    required this.subtitle,
    this.tag,
    this.canShowCriteria = false,
  });

  @override
  State<_FullImageViewer> createState() => _FullImageViewerState();
}

class _FullImageViewerState extends State<_FullImageViewer> {
  final TransformationController _zoom = TransformationController();
  bool _chromeVisible = true;

  @override
  void dispose() {
    _zoom.dispose();
    super.dispose();
  }

  void _onDoubleTap(TapDownDetails d) {
    final zoomed = _zoom.value.getMaxScaleOnAxis() > 1.01;
    if (zoomed) {
      _zoom.value = Matrix4.identity();
      return;
    }
    // 더블탭 지점을 중심으로 2.5배
    final p = d.localPosition;
    _zoom.value = Matrix4.identity()
      ..translateByDouble(-p.dx * 1.5, -p.dy * 1.5, 0, 1)
      ..scaleByDouble(2.5, 2.5, 1, 1);
  }

  @override
  Widget build(BuildContext context) {
    TapDownDetails? lastTap;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          GestureDetector(
            onTap: () => setState(() => _chromeVisible = !_chromeVisible),
            onDoubleTapDown: (d) => lastTap = d,
            onDoubleTap: () {
              if (lastTap != null) _onDoubleTap(lastTap!);
            },
            child: InteractiveViewer(
              transformationController: _zoom,
              minScale: 1,
              maxScale: 5,
              child: Center(
                child: Image.file(widget.file, fit: BoxFit.contain),
              ),
            ),
          ),
          // 상단 크롬 — 닫기 + 제목
          AnimatedOpacity(
            opacity: _chromeVisible ? 1 : 0,
            duration: const Duration(milliseconds: 180),
            child: IgnorePointer(
              ignoring: !_chromeVisible,
              child: Align(
                alignment: Alignment.topCenter,
                child: Container(
                  padding: EdgeInsets.fromLTRB(
                      16, MediaQuery.viewPaddingOf(context).top + 8, 16, 14),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Colors.black.withValues(alpha: 0.6),
                        Colors.transparent,
                      ],
                    ),
                  ),
                  child: Row(
                    children: [
                      Material(
                        color: Colors.white.withValues(alpha: 0.12),
                        shape: const CircleBorder(),
                        child: InkWell(
                          customBorder: const CircleBorder(),
                          onTap: () => Navigator.of(context).pop(),
                          child: const SizedBox(
                            width: 38,
                            height: 38,
                            child: Icon(Icons.close,
                                size: 18, color: kNeutral100),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(widget.title,
                                style: const TextStyle(
                                    fontSize: 15,
                                    fontWeight: FontWeight.w700,
                                    color: kNeutral100)),
                            Text(widget.subtitle,
                                style: TextStyle(
                                    fontSize: 11.5,
                                    color: Colors.white.withValues(alpha: 0.7))),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          // 하단 — 태그 + 액션(배출 기준 보기 · 삭제) + 힌트
          AnimatedOpacity(
            opacity: _chromeVisible ? 1 : 0,
            duration: const Duration(milliseconds: 180),
            child: IgnorePointer(
              ignoring: !_chromeVisible,
              child: Align(
                alignment: Alignment.bottomCenter,
                child: Container(
                  padding: EdgeInsets.fromLTRB(20, 24, 20,
                      MediaQuery.viewPaddingOf(context).bottom + 16),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.bottomCenter,
                      end: Alignment.topCenter,
                      colors: [
                        Colors.black.withValues(alpha: 0.75),
                        Colors.transparent,
                      ],
                    ),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (widget.tag != null)
                        Align(
                          alignment: Alignment.centerLeft,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 9, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.white.withValues(alpha: 0.14),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Text(
                              widget.tag!.$1,
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: widget.tag!.$2 ? kAccent200 : kNeutral100,
                              ),
                            ),
                          ),
                        ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: Material(
                              color: Colors.white.withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(14),
                              child: InkWell(
                                borderRadius: BorderRadius.circular(14),
                                onTap: () => Navigator.of(context).pop('delete'),
                                child: const SizedBox(
                                  height: 48,
                                  child: Center(
                                    child: Text('삭제',
                                        style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.w600,
                                            color: kNeutral100)),
                                  ),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            flex: 2,
                            child: Material(
                              color: widget.canShowCriteria
                                  ? kAccent700
                                  : Colors.white.withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(14),
                              child: InkWell(
                                borderRadius: BorderRadius.circular(14),
                                onTap: widget.canShowCriteria
                                    ? () => Navigator.of(context).pop('criteria')
                                    : null,
                                child: const SizedBox(
                                  height: 48,
                                  child: Center(
                                    child: Text('배출 기준 보기',
                                        style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.w700,
                                            color: kNeutral100)),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Center(
                        child: Text(
                          '두 손가락으로 확대 · 더블탭 확대/원래대로 · 탭하면 정보 숨김',
                          style: TextStyle(
                              fontSize: 11,
                              color: Colors.white.withValues(alpha: 0.55)),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
