import 'package:flutter/material.dart';

import '../../core/di/app_scope.dart';
import '../../data/collection_schedule.dart';
import '../../data/haptics.dart';
import '../../data/settings_store.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import '../../widgets/animated_entry.dart';
import '../../widgets/region_picker.dart';
import '../schedule/collection_schedule_screen.dart';
import 'widgets/home_widgets.dart';
import 'widgets/how_sheet.dart';

class HomeScreen extends StatefulWidget {
  /// 하단 내비게이션 탭 전환 (MainShell이 주입).
  final VoidCallback? onScanTap;
  final ValueChanged<String>? onSearchTap; // 통합 검색 탭 (필터 지정)
  final VoidCallback? onHistoryTap;
  final VoidCallback? onSettingsTap;

  const HomeScreen({
    super.key,
    this.onScanTap,
    this.onSearchTap,
    this.onHistoryTap,
    this.onSettingsTap,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}


class _HomeScreenState extends State<HomeScreen> {
  final SettingsStore _settings = AppScope.settings;
  final ScrollController _scrollController = ScrollController();

  bool _showScrollTop = false;
  String _currentApiUrl = '';
  bool _isTestMode = false;
  (String, String)? _region;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    _refreshState();
    _maybePromptRegion();
  }

  /// 첫 진입 시 지역 선택 안내 — 지자체 조례별 배출 기준 매핑용 (시나리오).
  /// 스킵하면 재노출하지 않음 (설정에서 언제든 변경 가능).
  Future<void> _maybePromptRegion() async {
    if (await _settings.getRegion() != null) return;
    if (await _settings.isRegionPromptShown()) return;
    await _settings.setRegionPromptShown();
    if (!mounted) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) showRegionPicker(context);
    });
  }

  @override
  void dispose() {
    _scrollController.removeListener(_onScroll);
    _scrollController.dispose();
    super.dispose();
  }

  void _onScroll() {
    final show = _scrollController.offset > 200;
    if (show != _showScrollTop) setState(() => _showScrollTop = show);
  }

  Future<void> _refreshState() async {
    final url = await _settings.getApiUrl();
    final region = await _settings.getRegion();
    if (!mounted) return;
    setState(() {
      _currentApiUrl = url;
      _isTestMode = url == SettingsStore.testModeApiUrl;
      _region = region;
    });
  }

  Future<void> _changeRegion() async {
    Haptics.selection();
    final picked = await showRegionPicker(context);
    if (picked != null && mounted) setState(() => _region = picked);
  }

  /// 시안 우측 상단 토글 — 라이트/다크 즉시 전환 (설정에도 영구 반영).
  void _toggleTheme() {
    Haptics.selection();
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final mode = isDark ? ThemeMode.light : ThemeMode.dark;
    appThemeMode.value = mode;
    _settings.setThemeMode(mode);
  }

  /// 통합 검색 탭으로 전환 (시안 7a: 검색창·스코프 칩이 진입점).
  void _openSearch({String filter = '전체'}) {
    Haptics.selection();
    widget.onSearchTap?.call(filter);
  }

  /// 수거일 안내 화면 (시안 4a).
  void _openSchedule() {
    Haptics.selection();
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => const CollectionScheduleScreen(),
      ),
    );
  }

  /// 헤더 ⓘ — 작동 원리 바텀시트 (시안 8번 모달).
  void _openHow() {
    Haptics.selection();
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (_) => const HowSheet(),
    );
  }

  @override
  Widget build(BuildContext context) {
    // 주거 형태가 바뀌면(설정·온보딩) 수거 일정 UI 노출을 즉시 갱신
    return ValueListenableBuilder<HousingType?>(
      valueListenable: appHousingType,
      builder: (context, housing, _) => ValueListenableBuilder<List<int>>(
        valueListenable: appPickupWeekdays,
        builder: (context, _, _) => _buildBody(context, housing: housing),
      ),
    );
  }

  /// 헤드라인 아래 한 줄 — 주거 형태·지역 설정에 따라 실용 정보.
  String _subline(HousingType? housing, int todayIdx) {
    if (housing == HousingType.apartment) {
      return '단지 분리배출장 · 시간 제약 없이 상시 배출';
    }
    if (_region == null) return '지역을 설정하면 우리 동네 수거일도 알려드려요';
    final today = effectiveWeekSchedule()[todayIdx];
    if (today != PickupKind.none) {
      return '오늘은 ${today.shortLabel} 버리는 날(예시) · $kPickupTimeText';
    }
    for (var i = 1; i < 7; i++) {
      final idx = (todayIdx + i) % 7;
      if (effectiveWeekSchedule()[idx] != PickupKind.none) {
        return '오늘은 배출 없는 날 · 다음 수거일 ${kDayNames[idx]}요일';
      }
    }
    return '오늘은 배출 없는 날';
  }

  Widget _buildBody(BuildContext context, {required HousingType? housing}) {
    final showSchedule = housing != HousingType.apartment;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final t = DsTokens(isDark);
    final now = DateTime.now();
    final todayIdx = now.weekday - 1;

    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: Stack(
          children: [
            SingleChildScrollView(
              controller: _scrollController,
              // 하단 여백은 바 위로 떠 있는 중앙 촬영 버튼(약 30px)까지 고려
              padding: EdgeInsets.fromLTRB(
                20,
                kSpaceL,
                20,
                kSpaceXXL + 16 + MediaQuery.viewPaddingOf(context).bottom,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 헤더 — 지역·날짜 + ⓘ + 테마 토글
                  AnimatedEntry(
                    child: Row(
                      children: [
                        Expanded(
                          child: InkWell(
                            borderRadius: BorderRadius.circular(kRadiusSmall),
                            onTap: _changeRegion,
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.place_outlined,
                                    size: 15, color: t.muted),
                                const SizedBox(width: 6),
                                Flexible(
                                  child: Text(
                                    '${_region?.$2 ?? '지역 설정'} · '
                                    '${now.month}월 ${now.day}일 '
                                    '(${kDayNames[todayIdx]})',
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      color: t.muted2,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                        HeaderIconButton(
                          tokens: t,
                          onTap: _openHow,
                          dimmed: true,
                          semanticLabel: '스마트 촬영 안내',
                          child: Icon(Icons.info_outline,
                              size: 16, color: t.muted2),
                        ),
                        const SizedBox(width: 8),
                        HeaderIconButton(
                          tokens: t,
                          onTap: _toggleTheme,
                          semanticLabel: isDark ? '라이트 모드로 전환' : '다크 모드로 전환',
                          child: Icon(
                            isDark
                                ? Icons.light_mode_outlined
                                : Icons.dark_mode_outlined,
                            size: 17,
                            color: t.accentChipText,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 28),
                  // 헤드라인 — 오늘의 배출 품목
                  AnimatedEntry(
                    index: 1,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // 브랜드 카피 — 주거 형태와 무관하게 고정, 강조 1단어
                        Text.rich(
                          TextSpan(
                            children: [
                              TextSpan(
                                text: '사진 한 장',
                                style: TextStyle(color: t.accentStrong),
                              ),
                              const TextSpan(text: '으로\n분리수거 해봐요'),
                            ],
                          ),
                          style: const TextStyle(
                            fontSize: 40,
                            fontWeight: FontWeight.w600,
                            height: 1.08,
                            letterSpacing: -0.4,
                          ),
                        ),
                        const SizedBox(height: 10),
                        // 실용 정보 한 줄 — 주택: 오늘 수거 품목 / 아파트: 상시 배출
                        Text(
                          _subline(housing, todayIdx),
                          style: TextStyle(fontSize: 12, color: t.muted),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                  // 통합 검색 진입창
                  AnimatedEntry(
                    index: 2,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Material(
                          color: Colors.transparent,
                          child: InkWell(
                            borderRadius: BorderRadius.circular(20),
                            onTap: _openSearch,
                            child: Ink(
                              height: 62,
                              padding: const EdgeInsets.fromLTRB(20, 0, 20, 0),
                              decoration: BoxDecoration(
                                color: t.surface,
                                border: Border.all(
                                  color: isDark ? kAccent700 : kAccent400,
                                  width: 1.5,
                                ),
                                borderRadius: BorderRadius.circular(20),
                                boxShadow: [
                                  BoxShadow(
                                    color: kInkCardShadow
                                        .withValues(alpha: 0.14),
                                    offset: const Offset(0, 1),
                                    blurRadius: 2,
                                  ),
                                ],
                              ),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      '그린가이드의 모든 것을 검색할 수 있어요',
                                      style: TextStyle(
                                        fontSize: 16,
                                        color: t.muted,
                                      ),
                                    ),
                                  ),
                                  Icon(Icons.search,
                                      size: 22, color: t.accentStrong),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                  // 오늘의 팁 — 패턴 배너 카드
                  AnimatedEntry(
                    index: 3,
                    child: TipCard(tokens: t),
                  ),
                  // 우리 동네 분리수거 일정 — 아파트(상시 배출)면 숨김
                  if (showSchedule) ...[
                  const SizedBox(height: 32),
                  AnimatedEntry(
                    index: 4,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              '우리 동네 분리수거 일정',
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 0.44,
                                color: t.muted,
                              ),
                            ),
                            const Spacer(),
                            InkWell(
                              borderRadius:
                                  BorderRadius.circular(kRadiusSmall),
                              onTap: _openSchedule,
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Text(
                                    '수거일 안내',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                      color: t.muted,
                                    ),
                                  ),
                                  Icon(Icons.chevron_right,
                                      size: 16, color: t.muted),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 14),
                        InkWell(
                          borderRadius: BorderRadius.circular(kRadiusMedium),
                          onTap: _openSchedule,
                          child: WeekStrip(tokens: t, todayIdx: todayIdx),
                        ),
                      ],
                    ),
                  ),
                  ],
                  // API 힌트는 개발용 — 테스트 모드에서만 노출
                  if (_isTestMode) ...[
                    const SizedBox(height: kSpaceXL),
                    AnimatedEntry(
                      index: 5,
                      child: HintCard(
                        currentApiUrl: _currentApiUrl,
                        isTestMode: _isTestMode,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            if (_showScrollTop)
              Positioned(
                bottom: kSpaceL + MediaQuery.viewPaddingOf(context).bottom,
                right: kSpaceL,
                child: FloatingActionButton.small(
                  onPressed: () {
                    Haptics.selection();
                    _scrollController.animateTo(
                      0,
                      duration: kPageTransitionDuration,
                      curve: kPageTransitionCurve,
                    );
                  },
                  tooltip: '맨 위로',
                  child: const Icon(Icons.keyboard_arrow_up),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
