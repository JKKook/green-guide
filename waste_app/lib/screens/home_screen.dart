import 'package:flutter/material.dart';

import '../core/di/app_scope.dart';
import '../core/ui/ds_card.dart';
import '../data/collection_schedule.dart';
import '../data/haptics.dart';
import '../data/settings_store.dart';
import '../data/tips.dart';
import '../features/schedule/collection_schedule_screen.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/animated_entry.dart';
import '../widgets/region_picker.dart';

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
      builder: (_) => const _HowSheet(),
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
                        _HeaderIconButton(
                          tokens: t,
                          onTap: _openHow,
                          dimmed: true,
                          semanticLabel: '스마트 촬영 안내',
                          child: Icon(Icons.info_outline,
                              size: 16, color: t.muted2),
                        ),
                        const SizedBox(width: 8),
                        _HeaderIconButton(
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
                    child: _TipCard(tokens: t),
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
                          child: _WeekStrip(tokens: t, todayIdx: todayIdx),
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
                      child: _HintCard(
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

/// 헤더 우측 38px 원형 아이콘 버튼.
class _HeaderIconButton extends StatelessWidget {
  final DsTokens tokens;
  final VoidCallback onTap;
  final bool dimmed;
  final Widget child;

  /// 아이콘만 있는 버튼이라 스크린리더용 이름이 필요하다.
  final String semanticLabel;
  const _HeaderIconButton({
    required this.tokens,
    required this.onTap,
    this.dimmed = false,
    required this.semanticLabel,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: semanticLabel,
      child: Opacity(
      opacity: dimmed ? 0.6 : 1,
      child: Material(
        color: tokens.surface,
        shape: CircleBorder(side: BorderSide(color: tokens.border)),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onTap,
          child: SizedBox(
            width: 38,
            height: 38,
            child: Center(child: child),
          ),
        ),
      ),
      ),
    );
  }
}


/// 오늘의 팁 카드 — 블레이드 스트로크 패턴 배너 + 일별 팁.
class _TipCard extends StatelessWidget {
  final DsTokens tokens;
  const _TipCard({required this.tokens});

  @override
  Widget build(BuildContext context) {
    final tip = todayTip();

    return DsCard(
      elevated: true,
      radius: 24,
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 배너 — 13번 패턴 라이브러리 에셋, 팁이 바뀌는 날마다 교체
          SizedBox(
            height: 150,
            width: double.infinity,
            child: ColorFiltered(
              // 다크 모드 — 라이트 팔레트 배너를 살짝 가라앉혀 대비 유지
              colorFilter: tokens.dark
                  ? const ColorFilter.mode(Color(0xFF9AA3B0), BlendMode.modulate)
                  : const ColorFilter.mode(Colors.transparent, BlendMode.dst),
              child: Image.asset(
                todayTipBanner(),
                fit: BoxFit.cover,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '오늘의 팁',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 5),
                Text(
                  tip,
                  style: TextStyle(
                    fontSize: 12.5,
                    height: 1.55,
                    color: tokens.muted2,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}


/// 주간 수거 스트립 — 오늘부터 7일, 오늘 칩 강조.
class _WeekStrip extends StatelessWidget {
  final DsTokens tokens;
  final int todayIdx;
  const _WeekStrip({required this.tokens, required this.todayIdx});

  Color? _dotColor(PickupKind p) => switch (p) {
        PickupKind.plasticVinyl => kAccent500,
        PickupKind.paperBox => kAccent2400,
        PickupKind.general => tokens.faint,
        PickupKind.none => null,
      };

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        for (var i = 0; i < 7; i++) ...[
          if (i > 0) const SizedBox(width: 7),
          Expanded(
            child: Builder(builder: (context) {
              final dayIdx = (todayIdx + i) % 7;
              final isToday = i == 0;
              final dot = _dotColor(effectiveWeekSchedule()[dayIdx]);
              return Container(
                padding: const EdgeInsets.fromLTRB(0, 12, 0, 11),
                decoration: BoxDecoration(
                  color: isToday ? kAccent700 : tokens.surface,
                  border:
                      isToday ? null : Border.all(color: tokens.border),
                  borderRadius: BorderRadius.circular(kRadiusMedium),
                ),
                child: Column(
                  children: [
                    Text(
                      kDayNames[dayIdx],
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight:
                            isToday ? FontWeight.w700 : FontWeight.w600,
                        color: isToday ? kNeutral100 : tokens.muted,
                      ),
                    ),
                    const SizedBox(height: 7),
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isToday ? kNeutral100 : dot,
                      ),
                    ),
                  ],
                ),
              );
            }),
          ),
        ],
      ],
    );
  }
}


/// 작동 원리 바텀시트 — 3스텝 + 조건 단계 + 피드백 안내 + 확인.
class _HowSheet extends StatelessWidget {
  const _HowSheet();

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


class _HintCard extends StatelessWidget {
  final String currentApiUrl;
  final bool isTestMode;
  const _HintCard({required this.currentApiUrl, required this.isTestMode});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final bgColor = isTestMode ? cs.primaryContainer : cs.surfaceContainerHigh;
    final fgColor = isTestMode ? cs.onPrimaryContainer : cs.onSurfaceVariant;
    return Container(
      padding: const EdgeInsets.all(kSpaceM),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(kRadiusMedium),
      ),
      child: Row(
        children: [
          Icon(
            isTestMode ? Icons.usb : Icons.cloud_outlined,
            size: 18,
            color: fgColor,
          ),
          const SizedBox(width: kSpaceS),
          Expanded(
            child: Text(
              isTestMode
                  ? '테스트 모드 — adb reverse 로 PC API 연결'
                  : 'API: $currentApiUrl',
              style: TextStyle(fontSize: 12, color: fgColor),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
