import 'package:flutter/material.dart';

import '../../data/haptics.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import '../capture/capture_entry_sheet.dart';
import '../history/history_screen.dart';
import '../home/home_screen.dart';
import '../search/unified_search_screen.dart';
import '../settings/settings_screen.dart';

/// 하단 내비게이션 셸 — 홈 · 검색 · 스마트 촬영(중앙) · 기록 · 설정.
/// 시안 8a 의 라운드 엣지 바: 위 모서리 26px + 중앙에 떠 있는
/// 그라디언트 원형 촬영 버튼. 검색은 통합 검색 화면, 촬영은 진입 시트(16a).
class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  // 탭 인덱스 — 0 홈 · 1 검색 · 2 기록 · 3 설정
  int _tab = 0;
  String _searchFilter = '전체';
  int _searchToken = 0;

  void _selectTab(int i) {
    Haptics.selection();
    setState(() => _tab = i);
  }

  /// 통합 검색 탭으로 — 홈 검색창·스코프 칩·하단바 검색 공용 (시안 7a→7b).
  void _openSearch({String filter = '전체'}) {
    Haptics.selection();
    setState(() {
      _tab = 1;
      _searchFilter = filter;
      _searchToken++;
    });
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final barColor = t.surface;

    return Scaffold(
      body: IndexedStack(
        index: _tab,
        children: [
          HomeScreen(
            onScanTap: () => showCaptureEntrySheet(context),
            onSearchTap: (filter) => _openSearch(filter: filter),
            onHistoryTap: () => setState(() => _tab = 2),
            onSettingsTap: () => setState(() => _tab = 3),
          ),
          UnifiedSearchScreen(
            initialFilter: _searchFilter,
            activationToken: _searchToken,
            onBack: () => _selectTab(0),
            onOpenHistory: () => setState(() => _tab = 2),
            onOpenSettings: () => setState(() => _tab = 3),
          ),
          const HistoryScreen(),
          const SettingsScreen(),
        ],
      ),
      // 중앙 촬영 버튼 — 바 위로 반쯤 떠 있는 그라디언트 원 (시안 8a)
      floatingActionButton: SizedBox(
        width: 62,
        height: 62,
        child: Material(
          color: Colors.transparent,
          shape: CircleBorder(
            side: BorderSide(color: barColor, width: 4),
          ),
          elevation: 5,
          child: Ink(
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [kAccent500, kAccent800],
              ),
            ),
            child: Semantics(
              button: true,
              label: '사진으로 분리배출 확인하기',
              child: InkWell(
                customBorder: const CircleBorder(),
                onTap: () => showCaptureEntrySheet(context),
                child: const Icon(
                  Icons.photo_camera_outlined,
                  color: kNeutral100,
                  size: 27,
                ),
              ),
            ),
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: barColor,
          border: Border.all(color: t.border),
          borderRadius:
              const BorderRadius.vertical(top: Radius.circular(26)),
          boxShadow: [
            BoxShadow(
              color: kInkShadow
                  .withValues(alpha: t.dark ? 0.3 : 0.08),
              offset: const Offset(0, -4),
              blurRadius: 14,
            ),
          ],
        ),
        padding: EdgeInsets.only(
          top: 10,
          bottom: MediaQuery.viewPaddingOf(context).bottom + 6,
        ),
        child: Row(
          children: [
            _NavItem(
              icon: Icons.home_outlined,
              label: '홈',
              selected: _tab == 0,
              onTap: () => _selectTab(0),
            ),
            _NavItem(
              icon: Icons.search,
              label: '검색',
              selected: _tab == 1,
              onTap: _openSearch,
            ),
            // 중앙 슬롯 — FAB 가 위를 차지하고 라벨만 바에 표시
            // (다른 항목의 아이콘 높이만큼 띄워 라벨 라인을 맞춘다)
            Expanded(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(height: 27),
                  Text(
                    '스마트 촬영',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: t.dark ? kAccent300 : kAccent900,
                    ),
                  ),
                ],
              ),
            ),
            _NavItem(
              icon: Icons.schedule,
              label: '기록',
              selected: _tab == 2,
              onTap: () => _selectTab(2),
            ),
            _NavItem(
              icon: Icons.settings_outlined,
              label: '설정',
              selected: _tab == 3,
              onTap: () => _selectTab(3),
            ),
          ],
        ),
      ),
    );
  }
}

/// 하단 바 항목 — 아이콘 + 라벨 (선택 시 잉크 컬러).
class _NavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _NavItem({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final color = selected
        ? Theme.of(context).colorScheme.onSurface
        : t.faint;
    return Expanded(
      child: InkWell(
        borderRadius: BorderRadius.circular(kRadiusMedium),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 2),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 24, color: color),
              const SizedBox(height: 3),
              Text(
                label,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w600,
                  color: color,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
