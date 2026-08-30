/// 기간 선택 시트 (프리셋·달력).
library;

import 'package:flutter/material.dart';
import '../../../data/haptics.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';

/// 기간 선택 바텀시트 — 시안 15b: 시작/종료 + 프리셋 + 레인지 캘린더.
class RangeSheet extends StatefulWidget {
  final DateTimeRange? initial;
  const RangeSheet({super.key, this.initial});

  @override
  State<RangeSheet> createState() => _RangeSheetState();
}


class _RangeSheetState extends State<RangeSheet> {
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
                  color: t.handle,
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
                    padding: const EdgeInsets.all(kSpaceXS),
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
                      borderRadius: BorderRadius.circular(kRadiusSmall),
                      onTap: () => _applyPreset(p),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 11, vertical: 6),
                        decoration: BoxDecoration(
                          color: _preset == p ? kAccent700 : t.surface,
                          border: _preset == p
                              ? null
                              : Border.all(color: t.border),
                          borderRadius: BorderRadius.circular(kRadiusSmall),
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
                    padding: const EdgeInsets.all(kSpaceXS),
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
                    padding: const EdgeInsets.all(kSpaceXS),
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
                    borderRadius: BorderRadius.circular(kRadiusMedium),
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
                              t.handle,
                        ),
                        borderRadius: BorderRadius.circular(kRadiusMedium),
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
                    borderRadius: BorderRadius.circular(kRadiusMedium),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(kRadiusMedium),
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
