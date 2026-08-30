import 'package:flutter/material.dart';

import '../core/di/app_scope.dart';
import '../data/collection_schedule.dart';
import '../data/haptics.dart';
import '../data/history_repository.dart';
import '../data/settings_store.dart';
import '../data/tips.dart';
import '../data/waste_info.dart';
import '../features/schedule/collection_reminders_screen.dart';
import '../features/schedule/collection_schedule_screen.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/criteria_sheet.dart';

/// 통합 검색 — 시안 7b: 품목·일정·기록·설정을 한 검색창에서.
class UnifiedSearchScreen extends StatefulWidget {
  /// 초기 필터 — 전체 · 품목 · 일정 · 기록 · 설정.
  final String initialFilter;

  /// 탭으로 다시 활성화될 때마다 바뀌는 토큰 — 필터 재적용 + 입력창 포커스.
  final int activationToken;

  /// 헤더 뒤로가기 (홈 탭 복귀). null 이면 뒤로가기 숨김.
  final VoidCallback? onBack;

  /// 결과 탭 이동 (MainShell 탭 전환).
  final VoidCallback? onOpenHistory;
  final VoidCallback? onOpenSettings;

  const UnifiedSearchScreen({
    super.key,
    this.initialFilter = '전체',
    this.activationToken = 0,
    this.onBack,
    this.onOpenHistory,
    this.onOpenSettings,
  });

  @override
  State<UnifiedSearchScreen> createState() => _UnifiedSearchScreenState();
}

class _UnifiedSearchScreenState extends State<UnifiedSearchScreen> {
  static const _allFilters = ['전체', '품목', '일정', '기록', '설정'];

  /// 아파트(상시 배출)면 '일정' 필터·그룹을 숨긴다.
  List<String> get _filters => showsCollectionSchedule
      ? _allFilters
      : [for (final f in _allFilters) if (f != '일정') f];

  final TextEditingController _controller = TextEditingController();
  final FocusNode _focus = FocusNode();
  final HistoryRepository _history = AppScope.history;
  final SettingsStore _settings = AppScope.settings;

  late String _filter = _filters.contains(widget.initialFilter)
      ? widget.initialFilter
      : '전체';
  String _query = '';
  List<HistoryEntry> _entries = [];
  (String, String)? _region;

  @override
  void initState() {
    super.initState();
    _load();
    historyRevision.addListener(_load);
  }

  @override
  void didUpdateWidget(covariant UnifiedSearchScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    // 홈 검색창·스코프 칩에서 다시 진입 — 필터 갱신 + 최신 기록 + 포커스
    if (oldWidget.activationToken != widget.activationToken) {
      setState(() {
        _filter = _filters.contains(widget.initialFilter)
            ? widget.initialFilter
            : '전체';
      });
      _load();
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _focus.requestFocus();
      });
    }
  }

  Future<void> _load() async {
    final entries = await _history.recent();
    final region = await _settings.getRegion();
    if (!mounted) return;
    setState(() {
      _entries = entries;
      _region = region;
    });
  }

  @override
  void dispose() {
    historyRevision.removeListener(_load);
    _controller.dispose();
    _focus.dispose();
    super.dispose();
  }

  bool _show(String group) => _filter == '전체' || _filter == group;

  // ── 그룹별 매칭 ────────────────────────────────────────────────

  /// 자주 찾는 품목 — 검색어 없이 '전체' 를 볼 때 앞세우는 기본 목록.
  static const _featuredItems = ['styrofoam', 'paper_pack', 'pet', 'battery_guide'];

  List<WasteInfo> get _itemResults {
    final q = _query.trim();
    final seen = <String>{};
    final all = [
      ...WasteClassRegistry.all
          .where((c) => c.classKey != 'non_object' && c.classKey != 'etc'),
      ...kExtraGuides,
    ].where((c) => seen.add(c.displayName)).toList();
    if (q.isEmpty) {
      // 브라우즈 — 품목 필터면 전체 목록, '전체' 면 자주 찾는 품목만
      if (_filter == '품목') return all;
      return [
        for (final key in _featuredItems)
          ?all.where((c) => c.classKey == key).firstOrNull,
      ];
    }
    return all
        .where((c) => c.displayName.contains(q) || c.summary.contains(q))
        .take(6)
        .toList();
  }

  /// 수거 일정 매칭 — 품목 라벨 또는 일정 관련 키워드.
  List<PickupKind> get _scheduleResults {
    if (!showsCollectionSchedule) return const [];
    final q = _query.trim();
    const scheduleWords = ['수거', '일정', '배출', '요일'];
    final kinds = {
      for (final k in effectiveWeekSchedule())
        if (k != PickupKind.none) k,
    }.toList();
    if (q.isEmpty) return kinds; // 브라우즈 — 수거 품목별 일정 전체
    if (scheduleWords.any((w) => q.contains(w) || w.contains(q))) {
      return kinds;
    }
    return kinds
        .where((k) =>
            k.fullLabel.contains(q) ||
            k.shortLabel.contains(q) ||
            (k == PickupKind.plasticVinyl && '재활용'.contains(q)))
        .toList();
  }

  List<HistoryEntry> get _recordResults {
    final q = _query.trim();
    if (q.isEmpty) {
      // 브라우즈 — 최근 기록 (기록 필터면 더 많이)
      return _entries.take(_filter == '기록' ? 10 : 3).toList();
    }
    return _entries.where((e) {
      final fine = infoFor(e.predictedClass)?.displayName ?? '';
      final coarseKey = kFineToCoarse[e.predictedClass] ?? e.predictedClass;
      final coarse = infoFor(coarseKey)?.displayName ?? '';
      return fine.contains(q) || coarse.contains(q);
    }).take(5).toList();
  }

  List<String> get _tipResults {
    final q = _query.trim();
    if (q.isEmpty) return [todayTip()]; // 브라우즈 — 오늘의 팁
    return kDailyTips.where((t) => t.contains(q)).take(3).toList();
  }

  /// 설정 진입점 매칭 — (제목, 부제, 액션 id, 아이콘).
  List<(String, String, String, IconData)> get _settingResults {
    final q = _query.trim();
    const items = [
      ('수거일 알림 설정', '설정 · 알림 발송은 준비 중', '알림 수거 리마인더', 'reminders',
          Icons.notifications_none),
      ('내 동네 설정', '설정 · 지역별 배출 기준', '지역 동네 조례', 'settings',
          Icons.place_outlined),
      ('다크 모드', '설정 · 화면 테마', '다크 라이트 테마 화면', 'settings',
          Icons.dark_mode_outlined),
      ('햅틱 피드백', '설정 · 버튼·결과 진동', '햅틱 진동', 'settings',
          Icons.vibration),
    ];
    if (q.isEmpty) {
      // 브라우즈 — 설정 필터면 전체, '전체' 면 수거일 알림만 (아파트는 숨김)
      return [
        for (final (title, sub, _, action, icon) in items)
          if ((_filter == '설정' || action == 'reminders') &&
              (action != 'reminders' || showsCollectionSchedule))
            (title, sub, action, icon),
      ];
    }
    return [
      for (final (title, sub, keywords, action, icon) in items)
        if ((title.contains(q) || keywords.contains(q)) &&
            (action != 'reminders' || showsCollectionSchedule))
          (title, sub, action, icon),
    ];
  }

  // ── 액션 ──────────────────────────────────────────────────────

  void _openSchedule() {
    Haptics.selection();
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const CollectionScheduleScreen()),
    );
  }

  void _openSetting(String action) {
    Haptics.selection();
    if (action == 'reminders') {
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => const CollectionRemindersScreen()),
      );
      return;
    }
    widget.onOpenSettings?.call();
  }

  void _openHistoryTab() {
    Haptics.selection();
    widget.onOpenHistory?.call();
  }

  void _showTip(String tip) {
    Haptics.selection();
    final t = DsTokens.of(context);
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (_) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(kSpaceXL, 0, kSpaceXL, kSpaceXL),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '오늘의 팁',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.44,
                  color: t.accentChipText,
                ),
              ),
              const SizedBox(height: kSpaceS),
              Text(
                tip,
                style: const TextStyle(fontSize: 15, height: 1.6),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── UI ────────────────────────────────────────────────────────

  /// 검색어 하이라이트 — 시안: 일치 구간을 accent-700 로.
  Widget _highlight(String text, DsTokens t) {
    final q = _query.trim();
    final idx = q.isEmpty ? -1 : text.indexOf(q);
    if (idx < 0) {
      return Text(
        text,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(fontSize: 14.5, fontWeight: FontWeight.w600),
      );
    }
    return Text.rich(
      TextSpan(children: [
        TextSpan(text: text.substring(0, idx)),
        TextSpan(
          text: text.substring(idx, idx + q.length),
          style: TextStyle(color: t.accentStrong),
        ),
        TextSpan(text: text.substring(idx + q.length)),
      ]),
      maxLines: 1,
      overflow: TextOverflow.ellipsis,
      style: const TextStyle(fontSize: 14.5, fontWeight: FontWeight.w600),
    );
  }

  Widget _groupLabel(DsTokens t, String label) => Padding(
        padding: const EdgeInsets.only(top: 22),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.44,
            color: t.muted,
          ),
        ),
      );

  Widget _resultRow(
    DsTokens t, {
    required Widget icon,
    required bool accentTile,
    required Widget title,
    required String subtitle,
    required VoidCallback onTap,
    bool lastInGroup = false,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 13),
        decoration: BoxDecoration(
          border: lastInGroup
              ? null
              : Border(bottom: BorderSide(color: t.border)),
        ),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: accentTile ? t.accentChipBg : t.surface,
                border: accentTile ? null : Border.all(color: t.border),
                borderRadius: BorderRadius.circular(13),
              ),
              child: Center(child: icon),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  title,
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 12, color: t.muted),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right, size: 16, color: t.faint),
          ],
        ),
      ),
    );
  }

  /// 품목 행 부제 앞머리 — 재활용 / 일반쓰레기 (시안: "재활용 · 테이프·송장 제거 후 배출").
  String _itemTag(WasteInfo info) {
    final coarse = kFineToCoarse[info.classKey] ?? info.classKey;
    if (coarse == 'trash' || coarse == 'etc' || coarse == 'non_object') {
      return '일반쓰레기';
    }
    if (coarse == 'food_waste') return '음식물';
    return '재활용';
  }

  /// 수거 요일 요약 — '매주 화 · 금 저녁 (예시)'.
  /// 실데이터 연동 전이라 기본 예시 일정임을 함께 표시한다.
  String _scheduleDays(PickupKind kind) {
    final days = [
      for (var i = 0; i < 7; i++)
        if (effectiveWeekSchedule()[i] == kind) kDayNames[i],
    ];
    return '매주 ${days.join(' · ')} 저녁 (예시)';
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<HousingType?>(
      valueListenable: appHousingType,
      builder: (context, _, _) => _buildBody(context),
    );
  }

  Widget _buildBody(BuildContext context) {
    final t = DsTokens.of(context);
    final q = _query.trim();
    if (!_filters.contains(_filter)) _filter = '전체';
    final items = _show('품목') ? _itemResults : const <WasteInfo>[];
    final schedules = _show('일정') ? _scheduleResults : const <PickupKind>[];
    final records = _show('기록') ? _recordResults : const <HistoryEntry>[];
    final tips = _show('설정') ? _tipResults : const <String>[];
    final settings = _show('설정')
        ? _settingResults
        : const <(String, String, String, IconData)>[];
    final noResults = q.isNotEmpty &&
        items.isEmpty &&
        schedules.isEmpty &&
        records.isEmpty &&
        tips.isEmpty &&
        settings.isEmpty;

    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: ListView(
          keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
          padding: EdgeInsets.fromLTRB(
            20,
            kSpaceM,
            20,
            kSpaceXXL + 16 + MediaQuery.viewPaddingOf(context).bottom,
          ),
          children: [
            // 헤더 — 뒤로가기(홈) + 제목 (시안 7b)
            Padding(
              padding: const EdgeInsets.only(bottom: 14),
              child: Row(
                children: [
                  InkWell(
                    borderRadius: BorderRadius.circular(999),
                    onTap: () {
                      Haptics.selection();
                      if (widget.onBack != null) {
                        widget.onBack!();
                      } else {
                        Navigator.of(context).maybePop();
                      }
                    },
                    child: const Padding(
                      padding: EdgeInsets.fromLTRB(0, 4, 10, 4),
                      child: Icon(Icons.chevron_left,
                          size: 22, semanticLabel: '뒤로'),
                    ),
                  ),
                  const Text(
                    '통합 검색',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                ],
              ),
            ),
            // 검색 입력 — h54 · radius 18 · accent-500 보더
            Container(
              height: 54,
              padding: const EdgeInsets.only(left: 18, right: 8),
              decoration: BoxDecoration(
                color: t.surface,
                border: Border.all(color: kAccent500, width: 1.5),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      focusNode: _focus,
                      onChanged: (v) => setState(() => _query = v),
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.w600),
                      decoration: InputDecoration(
                        isDense: true,
                        border: InputBorder.none,
                        enabledBorder: InputBorder.none,
                        focusedBorder: InputBorder.none,
                        contentPadding: EdgeInsets.zero,
                        filled: false,
                        hintText: '그린가이드의 모든 것을 검색할 수 있어요',
                        hintStyle: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w400,
                          color: t.muted,
                        ),
                      ),
                    ),
                  ),
                  if (q.isEmpty)
                    Padding(
                      padding: const EdgeInsets.only(right: kSpaceS),
                      child: Icon(Icons.search, size: 20, color: t.faint),
                    )
                  else
                    IconButton(
                      tooltip: '검색어 지우기',
                      icon: Icon(Icons.cancel_outlined,
                          size: 20, color: t.faint),
                      onPressed: () {
                        Haptics.selection();
                        _controller.clear();
                        setState(() => _query = '');
                      },
                    ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            // 필터 칩
            Row(
              children: [
                for (final f in _filters)
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(14),
                      onTap: () {
                        Haptics.selection();
                        setState(() => _filter = f);
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 13, vertical: 7),
                        decoration: BoxDecoration(
                          color: _filter == f ? kAccent700 : t.surface,
                          border: _filter == f
                              ? null
                              : Border.all(color: t.border),
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Text(
                          f,
                          style: TextStyle(
                            fontSize: 11.5,
                            fontWeight: _filter == f
                                ? FontWeight.w700
                                : FontWeight.w600,
                            color:
                                _filter == f ? kNeutral100 : t.muted2,
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),

            if (noResults) ...[
              const SizedBox(height: 48),
              Center(
                child: Text(
                  '\'$q\' 검색 결과가 없어요\n촬영하면 AI가 바로 알려드려요',
                  textAlign: TextAlign.center,
                  style:
                      TextStyle(fontSize: 13, height: 1.6, color: t.muted2),
                ),
              ),
            ] else ...[
              // 품목 배출법
              if (items.isNotEmpty) ...[
                _groupLabel(t, '품목 배출법'),
                for (final (i, info) in items.indexed)
                  _resultRow(
                    t,
                    accentTile: true,
                    icon: Icon(Icons.view_in_ar_outlined,
                        size: 19, color: t.accentStrong),
                    title: _highlight(info.displayName, t),
                    subtitle: '${_itemTag(info)} · '
                        '${info.howTo.isNotEmpty ? info.howTo.first : (info.bin.isNotEmpty ? info.bin : info.summary)}',
                    onTap: () => showCriteriaSheet(context, info),
                    lastInGroup: i == items.length - 1,
                  ),
              ],
              // 수거 일정
              if (schedules.isNotEmpty) ...[
                _groupLabel(t, '수거 일정'),
                for (final (i, kind) in schedules.indexed)
                  _resultRow(
                    t,
                    accentTile: false,
                    icon: Icon(Icons.calendar_month_outlined,
                        size: 19, color: t.muted2),
                    title: _highlight('${kind.shortLabel} 수거일', t),
                    subtitle:
                        '${_region?.$2 ?? '지역 미설정'} · ${_scheduleDays(kind)}',
                    onTap: _openSchedule,
                    lastInGroup: i == schedules.length - 1,
                  ),
              ],
              // 내 기록
              if (records.isNotEmpty) ...[
                _groupLabel(t, '내 기록'),
                for (final (i, e) in records.indexed)
                  _resultRow(
                    t,
                    accentTile: false,
                    icon: Icon(Icons.schedule, size: 19, color: t.muted2),
                    title: _highlight(
                      '스마트 촬영 — '
                      '${infoFor(e.predictedClass)?.displayName ?? e.predictedClass}',
                      t,
                    ),
                    subtitle: '${e.createdAt.month}월 ${e.createdAt.day}일 · '
                        '${(kFineToCoarse[e.predictedClass] ?? e.predictedClass) == 'trash' ? '일반쓰레기' : '재활용'}로 분류됨',
                    onTap: _openHistoryTab,
                    lastInGroup: i == records.length - 1,
                  ),
              ],
              // 가이드 · 설정
              if (tips.isNotEmpty || settings.isNotEmpty) ...[
                _groupLabel(t, '가이드 · 설정'),
                for (final (i, tip) in tips.indexed)
                  _resultRow(
                    t,
                    accentTile: false,
                    icon: Icon(Icons.tips_and_updates_outlined,
                        size: 19, color: t.muted2),
                    title: _highlight('오늘의 팁 — $tip', t),
                    subtitle: '가이드 문서',
                    onTap: () => _showTip(tip),
                    lastInGroup:
                        settings.isEmpty && i == tips.length - 1,
                  ),
                for (final (i, (title, sub, action, icon)) in settings.indexed)
                  _resultRow(
                    t,
                    accentTile: false,
                    icon: Icon(icon, size: 19, color: t.muted2),
                    title: _highlight(title, t),
                    subtitle: sub,
                    onTap: () => _openSetting(action),
                    lastInGroup: i == settings.length - 1,
                  ),
              ],
            ],
          ],
        ),
      ),
    );
  }
}
