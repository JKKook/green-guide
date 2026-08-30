import 'package:flutter/material.dart';

import '../core/di/app_scope.dart';
import '../core/ui/ds_card.dart';
import '../data/collection_schedule.dart';
import '../data/haptics.dart';
import '../data/legal_terms.dart';
import '../data/region_data.dart';
import '../data/settings_store.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/korea_map.dart';
import '../widgets/region_picker.dart' show locateRegionFromGps;
import 'main_shell.dart';
import 'terms_screen.dart';

/// 첫 온보딩 — 시안 17: ① 이용 동의 → ② 지역(지도 → 시·군·구 시트)
/// → ③ 세대 구분 → ④ 주택·빌라면 분리 수거 설정 / 아파트면 마무리.
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final SettingsStore _settings = AppScope.settings;
  int _step = 0; // 0 동의 · 1 지역 · 2 수거 설정/마무리
  (String, String)? _region;
  HousingType _housing = HousingType.house;
  bool _alarmOptIn = false;

  Future<void> _onConsented({
    required bool alarmOptIn,
    required bool aiOptIn,
  }) async {
    Haptics.medium();
    await _settings.setConsentAccepted();
    await _settings.setCollectionAlarmOptIn(alarmOptIn);
    await _settings.setAiTrainingOptIn(aiOptIn);
    if (!mounted) return;
    setState(() {
      _alarmOptIn = alarmOptIn;
      _step = 1;
    });
  }

  /// 지역 단계 완료(선택 또는 나중에) → 세대 구분 시트 → 다음 단계.
  Future<void> _onRegionDone((String, String)? region) async {
    if (region != null) await _settings.setRegion(region.$1, region.$2);
    // 온보딩에서 이미 물었으므로 홈 첫 진입 때 같은 질문을 반복하지 않는다.
    await _settings.setRegionPromptShown();
    if (!mounted) return;
    _region = region;
    final housing = await showHousingTypeSheet(context, stepLabel: '2/3');
    if (!mounted) return;
    final picked = housing ?? HousingType.house;
    await _settings.setHousingType(picked);
    if (!mounted) return;
    setState(() {
      _housing = picked;
      _step = 2;
    });
  }

  Future<void> _finish() async {
    Haptics.medium();
    await _settings.setOnboardingDone();
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const MainShell()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final Widget body = switch (_step) {
      0 => _ConsentStep(onDone: _onConsented),
      1 => _RegionStep(onDone: _onRegionDone),
      _ => _housing == HousingType.house
          ? _PickupSetupStep(
              region: _region,
              alarmDefault: _alarmOptIn,
              onDone: _finish,
            )
          : _ApartmentFinishStep(region: _region, onDone: _finish),
    };
    return Scaffold(
      body: AnimatedSwitcher(
        duration: const Duration(milliseconds: 320),
        switchInCurve: Curves.easeOut,
        child: KeyedSubtree(key: ValueKey(_step), child: body),
      ),
    );
  }
}

// ─── 공용 조각 ──────────────────────────────────────────────────────────────

/// 주요 CTA — 52px · radius 16 · accent-700.
class _PrimaryButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final VoidCallback? onTap;
  const _PrimaryButton({required this.label, this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final enabled = onTap != null;
    return Material(
      color: enabled ? kAccent700 : t.border,
      borderRadius: BorderRadius.circular(kRadiusMedium),
      child: InkWell(
        borderRadius: BorderRadius.circular(kRadiusMedium),
        onTap: onTap,
        child: SizedBox(
          height: 52,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (icon != null) ...[
                Icon(icon, size: 18, color: enabled ? kNeutral100 : t.muted),
                const SizedBox(width: 9),
              ],
              Text(
                label,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  color: enabled ? kNeutral100 : t.muted,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 단계 표시 "1/3".
class _StepBadge extends StatelessWidget {
  final String label; // '1/3'
  const _StepBadge(this.label);

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final parts = label.split('/');
    return Text.rich(
      TextSpan(children: [
        TextSpan(text: parts.first),
        TextSpan(text: '/${parts.last}', style: TextStyle(color: t.faint)),
      ]),
      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: t.muted),
    );
  }
}

/// 시안의 바텀시트 카드 — 상단 radius 28 + 핸들.
class _SheetCard extends StatelessWidget {
  final Widget child;
  const _SheetCard({required this.child});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Theme.of(context).scaffoldBackgroundColor,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [
          BoxShadow(
            color: kInkShadow.withValues(alpha: 0.3),
            offset: const Offset(0, -10),
            blurRadius: 34,
          ),
        ],
      ),
      padding: EdgeInsets.fromLTRB(
          24, 14, 24, 24 + MediaQuery.viewPaddingOf(context).bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color: t.handle,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          child,
        ],
      ),
    );
  }
}

// ─── ① 이용 동의 (17a) ─────────────────────────────────────────────────────

class _ConsentStep extends StatefulWidget {
  final Future<void> Function({required bool alarmOptIn, required bool aiOptIn})
      onDone;
  const _ConsentStep({required this.onDone});

  @override
  State<_ConsentStep> createState() => _ConsentStepState();
}

class _ConsentStepState extends State<_ConsentStep> {
  /// 동의 항목 = 약관 문서 목록 (필수 3 · 선택 2) — 순서 고정.
  static final List<LegalDoc> _items = kLegalDocs;

  final List<bool> _checked = [false, false, false, false, false];
  bool _busy = false;

  bool get _allChecked => _checked.every((c) => c);
  bool get _requiredOk =>
      [for (final (i, it) in _items.indexed) if (it.required) _checked[i]]
          .every((c) => c);

  void _toggleAll() {
    Haptics.selection();
    final next = !_allChecked;
    setState(() {
      for (var i = 0; i < _checked.length; i++) {
        _checked[i] = next;
      }
    });
  }

  void _showDetail(int i) {
    Haptics.selection();
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => TermsDetailScreen(doc: _items[i])),
    );
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Stack(
      fit: StackFit.expand,
      children: [
        // 뒤 배경 — 브랜드 블록 (시안: blur + 55% opacity)
        Opacity(
          opacity: 0.55,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 88,
                height: 88,
                decoration: BoxDecoration(
                  color: kAccent700,
                  borderRadius: BorderRadius.circular(kRadiusXL),
                ),
                child: const Icon(Icons.recycling, size: 46, color: kNeutral100),
              ),
              const SizedBox(height: 16),
              const Text(
                '그린가이드',
                style: TextStyle(
                  fontSize: 34,
                  fontWeight: FontWeight.w600,
                  letterSpacing: -0.34,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                '사진 한 장으로 끝내는 분리배출',
                style: TextStyle(fontSize: 13, color: t.muted2),
              ),
              const SizedBox(height: 200),
            ],
          ),
        ),
        // 스크림 + 동의 시트
        Container(color: kInkShadow.withValues(alpha: 0.42)),
        Align(
          alignment: Alignment.bottomCenter,
          child: SingleChildScrollView(
            child: _SheetCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '그린가이드 이용 동의',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '서비스 시작을 위해 약관에 동의해주세요',
                    style: TextStyle(fontSize: 12.5, color: t.muted2),
                  ),
                  const SizedBox(height: 18),
                  // 전체 동의
                  InkWell(
                    borderRadius: BorderRadius.circular(kRadiusMedium),
                    onTap: _toggleAll,
                    child: Container(
                      padding: const EdgeInsets.fromLTRB(16, 15, 16, 15),
                      decoration: BoxDecoration(
                        color: t.accentChipBg,
                        border: Border.all(
                          color: _allChecked
                              ? (t.dark ? kAccent500 : kAccent400)
                              : t.border,
                          width: 1.5,
                        ),
                        borderRadius: BorderRadius.circular(kRadiusMedium),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              color: _allChecked ? kAccent700 : t.surface,
                              border: _allChecked
                                  ? null
                                  : Border.all(color: t.faint, width: 1.5),
                              shape: BoxShape.circle,
                            ),
                            child: _allChecked
                                ? const Icon(Icons.check,
                                    size: 14, color: kNeutral100)
                                : null,
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            '전체 동의',
                            style: TextStyle(
                                fontSize: 14.5, fontWeight: FontWeight.w700),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  for (final (i, item) in _items.indexed)
                    InkWell(
                      borderRadius: BorderRadius.circular(kRadiusSmall),
                      onTap: () {
                        Haptics.selection();
                        setState(() => _checked[i] = !_checked[i]);
                      },
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(kSpaceL, kSpaceM, kSpaceS, kSpaceM),
                        child: Row(
                          children: [
                            Icon(
                              Icons.check,
                              size: 17,
                              color: _checked[i]
                                  ? t.accentStrong
                                  : (t.dark
                                      ? const Color(0xFF5D5D60)
                                      : kNeutral300),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text.rich(
                                TextSpan(children: [
                                  TextSpan(
                                    text: item.required ? '[필수] ' : '[선택] ',
                                    style: item.required
                                        ? TextStyle(
                                            fontWeight: FontWeight.w700,
                                            color: t.accentChipText)
                                        : null,
                                  ),
                                  TextSpan(
                                      text: item.title
                                          .replaceAll(' (선택)', '')),
                                ]),
                                style: TextStyle(
                                  fontSize: 13,
                                  color: t.dark
                                      ? t.muted2
                                      : const Color(0xFF5D5D60),
                                ),
                              ),
                            ),
                            InkWell(
                              borderRadius: BorderRadius.circular(999),
                              onTap: () => _showDetail(i),
                              child: Padding(
                                padding: const EdgeInsets.all(6),
                                child: Icon(Icons.chevron_right,
                                    size: 15, color: t.faint),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(height: 14),
                  _PrimaryButton(
                    label: '동의하고 시작하기',
                    onTap: _requiredOk && !_busy
                        ? () async {
                            setState(() => _busy = true);
                            await widget.onDone(
                              alarmOptIn: _checked[3],
                              aiOptIn: _checked[4],
                            );
                          }
                        : null,
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ─── ② 지역 선택 (17b) + 시·군·구 시트 (17f) ─────────────────────────────────

class _RegionStep extends StatefulWidget {
  final ValueChanged<(String, String)?> onDone;
  const _RegionStep({required this.onDone});

  @override
  State<_RegionStep> createState() => _RegionStepState();
}

class _RegionStepState extends State<_RegionStep> {
  String? _sido;
  bool _locating = false;
  String? _gpsError;
  bool _done = false;

  Future<void> _useMyLocation() async {
    Haptics.selection();
    setState(() {
      _locating = true;
      _gpsError = null;
    });
    final result = await locateRegionFromGps();
    if (!mounted) return;
    setState(() => _locating = false);
    if (result.region == null) {
      setState(() => _gpsError = result.error);
      return;
    }
    _complete(result.region);
  }

  Future<void> _pickSigungu() async {
    final sido = _sido;
    if (sido == null) return;
    Haptics.selection();
    final picked = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      builder: (_) => _SigunguSheet(sido: sido),
    );
    if (picked == null || !mounted) return;
    _complete((sido, picked));
  }

  void _complete((String, String)? region) {
    if (_done) return;
    _done = true;
    widget.onDone(region);
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return SafeArea(
      bottom: false,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.place_outlined, size: 20, color: t.accentStrong),
                    const SizedBox(width: 8),
                    const Expanded(
                      child: Text(
                        '어느 지역에 사시나요?',
                        style:
                            TextStyle(fontSize: 26, fontWeight: FontWeight.w600),
                      ),
                    ),
                    const _StepBadge('1/3'),
                    const SizedBox(width: 10),
                    InkWell(
                      borderRadius: BorderRadius.circular(999),
                      onTap: () {
                        Haptics.selection();
                        _complete(null);
                      },
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 4),
                        child: Text(
                          '나중에',
                          style: TextStyle(
                            fontSize: 13.5,
                            fontWeight: FontWeight.w700,
                            color: t.muted2,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  '지역마다 분리배출 기준(조례)이 조금씩 달라요',
                  style: TextStyle(fontSize: 12.5, color: t.muted2),
                ),
                const SizedBox(height: 14),
                _PrimaryButton(
                  label: _locating ? '위치 확인 중...' : '내 위치로 설정',
                  icon: Icons.my_location,
                  onTap: _locating ? null : _useMyLocation,
                ),
                if (_gpsError != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    _gpsError!,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                        fontSize: 12,
                        color: Theme.of(context).colorScheme.error),
                  ),
                ],
                Padding(
                  padding: const EdgeInsets.fromLTRB(0, 14, 0, 6),
                  child: Row(
                    children: [
                      Expanded(child: Container(height: 1, color: t.border)),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: kSpaceM),
                        child: Text(
                          '또는 지도에서 선택',
                          style: TextStyle(
                            fontSize: 11.5,
                            fontWeight: FontWeight.w600,
                            color: t.muted,
                          ),
                        ),
                      ),
                      Expanded(child: Container(height: 1, color: t.border)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          // 지도 — 핀치 줌/팬
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(6, 4, 6, 0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(kRadiusMedium),
                child: InteractiveViewer(
                  maxScale: 6,
                  child: Center(
                    child: KoreaMap(
                      onSelect: (sido) {
                        Haptics.selection();
                        setState(() => _sido = sido);
                      },
                    ),
                  ),
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(0, 4, 0, 10),
            child: Text(
              '두 손가락으로 확대할 수 있어요',
              style: TextStyle(fontSize: 11.5, color: t.muted),
            ),
          ),
          // 하단 — 선택된 시·도 + 시·군·구 고르기
          Container(
            padding: EdgeInsets.fromLTRB(
                24, 16, 24, 16 + MediaQuery.viewPaddingOf(context).bottom),
            decoration: BoxDecoration(
              color: t.surface,
              border: Border(top: BorderSide(color: t.border)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    Container(
                      width: 20,
                      height: 20,
                      decoration: BoxDecoration(
                        color: _sido == null ? t.border : kAccent700,
                        borderRadius: BorderRadius.circular(6),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _sido == null ? '지도에서 시·도를 눌러주세요' : '$_sido 선택됨',
                            style: const TextStyle(
                                fontSize: 14.5, fontWeight: FontWeight.w700),
                          ),
                          Text(
                            '조례 기준 안내를 위해 시·군·구까지 골라주세요',
                            style: TextStyle(fontSize: 11.5, color: t.muted),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                _PrimaryButton(
                  label: '시·군·구 고르기',
                  onTap: _sido == null ? null : _pickSigungu,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// 시·군·구 선택 시트 — 조례 기준 단위까지만.
class _SigunguSheet extends StatefulWidget {
  final String sido;
  const _SigunguSheet({required this.sido});

  @override
  State<_SigunguSheet> createState() => _SigunguSheetState();
}

class _SigunguSheetState extends State<_SigunguSheet> {
  static const _collapsedCount = 12;
  String? _selected;
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final all = kKoreaRegions[widget.sido] ?? const <String>[];
    final visible =
        _expanded || all.length <= _collapsedCount ? all : all.take(_collapsedCount);
    // 시·도명 축약 — "경기도, 어디에 사세요?"
    final shortSido = widget.sido
        .replaceAll('특별자치도', '')
        .replaceAll('특별자치시', '')
        .replaceAll('특별시', '')
        .replaceAll('광역시', '');

    return SafeArea(
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(
            24, 8, 24, 24 + MediaQuery.viewInsetsOf(context).bottom),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: 20),
                decoration: BoxDecoration(
                  color: t.handle,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            Row(
              children: [
                Expanded(
                  child: Text(
                    '$shortSido, 어디에 사세요?',
                    style: const TextStyle(
                        fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                ),
                InkWell(
                  borderRadius: BorderRadius.circular(999),
                  onTap: () => Navigator.of(context).pop(),
                  child: Padding(
                    padding: const EdgeInsets.all(kSpaceXS),
                    child: Icon(Icons.close, size: 20, color: t.faint),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              '분리배출 기준은 시·군·구 조례로 정해져요 · 동까지는 몰라도 돼요',
              style: TextStyle(fontSize: 12.5, color: t.muted2),
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 7,
              runSpacing: 7,
              children: [
                for (final name in visible)
                  InkWell(
                    borderRadius: BorderRadius.circular(999),
                    onTap: () {
                      Haptics.selection();
                      setState(() => _selected = name);
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: _selected == name ? kAccent700 : t.surface,
                        border: _selected == name
                            ? null
                            : Border.all(color: t.border),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          if (_selected == name) ...[
                            const Icon(Icons.check,
                                size: 13, color: kNeutral100),
                            const SizedBox(width: 5),
                          ],
                          Text(
                            name,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: _selected == name
                                  ? FontWeight.w700
                                  : FontWeight.w600,
                              color: _selected == name
                                  ? kNeutral100
                                  : (t.dark
                                      ? t.muted2
                                      : const Color(0xFF5D5D60)),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                if (!_expanded && all.length > _collapsedCount)
                  InkWell(
                    borderRadius: BorderRadius.circular(999),
                    onTap: () {
                      Haptics.selection();
                      setState(() => _expanded = true);
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: t.handle,
                        ),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            '전체 ${all.length}개 시·군·구 보기',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              color: t.muted,
                            ),
                          ),
                          const SizedBox(width: 5),
                          Icon(Icons.expand_more, size: 13, color: t.muted),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 18),
            _PrimaryButton(
              label: _selected == null ? '시·군·구를 골라주세요' : '$_selected로 설정',
              onTap: _selected == null
                  ? null
                  : () => Navigator.of(context).pop(_selected),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.apartment_outlined, size: 13, color: t.muted),
                const SizedBox(width: 6),
                Flexible(
                  child: Text(
                    '${_selected ?? shortSido} 조례 기준으로 배출 방법과 수거 일정을 안내해요',
                    style: TextStyle(fontSize: 11.5, color: t.muted),
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

// ─── ③ 세대 구분 시트 (17c) — 설정 화면에서도 재사용 ──────────────────────────

/// 주거 형태 선택 시트. 온보딩(2/3 배지)·설정(배지 없음) 공용.
Future<HousingType?> showHousingTypeSheet(
  BuildContext context, {
  HousingType? current,
  String? stepLabel,
}) {
  return showModalBottomSheet<HousingType>(
    context: context,
    isDismissible: stepLabel == null,
    enableDrag: stepLabel == null,
    builder: (_) => _HousingTypeSheet(current: current, stepLabel: stepLabel),
  );
}

class _HousingTypeSheet extends StatefulWidget {
  final HousingType? current;
  final String? stepLabel;
  const _HousingTypeSheet({this.current, this.stepLabel});

  @override
  State<_HousingTypeSheet> createState() => _HousingTypeSheetState();
}

class _HousingTypeSheetState extends State<_HousingTypeSheet> {
  late HousingType _value = widget.current ?? HousingType.house;

  Widget _option({
    required HousingType type,
    required IconData icon,
    required String title,
    required String subtitle,
  }) {
    final t = DsTokens.of(context);
    final selected = _value == type;
    return InkWell(
      borderRadius: BorderRadius.circular(20),
      onTap: () {
        Haptics.selection();
        setState(() => _value = type);
      },
      child: Container(
        padding: const EdgeInsets.all(kSpaceL),
        decoration: BoxDecoration(
          color: selected ? t.accentChipBg : t.surface,
          border: Border.all(
            color: selected ? kAccent500 : t.border,
            width: selected ? 1.5 : 1,
          ),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Container(
              width: 46,
              height: 46,
              decoration: BoxDecoration(
                color: selected ? t.bannerBg : t.surface,
                border: Border.all(
                  color: selected
                      ? (t.accentChipBorder)
                      : t.border,
                ),
                borderRadius: BorderRadius.circular(kRadiusMedium),
              ),
              child: Icon(icon,
                  size: 22, color: selected ? t.accentChipText : t.muted2),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: const TextStyle(
                          fontSize: 15.5, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 2),
                  Text(subtitle,
                      style: TextStyle(fontSize: 12, color: t.muted2)),
                ],
              ),
            ),
            Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                color: selected ? kAccent700 : Colors.transparent,
                border: selected
                    ? null
                    : Border.all(
                        color: t.handle,
                        width: 1.5),
                shape: BoxShape.circle,
              ),
              child: selected
                  ? const Icon(Icons.check, size: 12, color: kNeutral100)
                  : null,
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return _SheetCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Expanded(
                child: Text(
                  '어떤 집에 사세요?',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                ),
              ),
              if (widget.stepLabel != null) _StepBadge(widget.stepLabel!),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '주거 형태에 따라 분리배출 방법이 달라져요',
            style: TextStyle(fontSize: 12.5, color: t.muted2),
          ),
          const SizedBox(height: 18),
          _option(
            type: HousingType.apartment,
            icon: Icons.apartment_outlined,
            title: '아파트 · 오피스텔',
            subtitle: '단지 내 분리배출장에 상시 배출',
          ),
          const SizedBox(height: 10),
          _option(
            type: HousingType.house,
            icon: Icons.home_outlined,
            title: '주택 · 빌라',
            subtitle: '동네 수거 요일에 맞춰 문 앞 배출',
          ),
          const SizedBox(height: 16),
          _PrimaryButton(
            label: '선택 완료',
            onTap: () => Navigator.of(context).pop(_value),
          ),
          const SizedBox(height: 12),
          Center(
            child: Text(
              '나중에 설정에서 변경할 수 있어요',
              style: TextStyle(fontSize: 11.5, color: t.muted),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── ④ 주택·빌라 — 분리 수거 설정 (17d) ──────────────────────────────────────

class _PickupSetupStep extends StatefulWidget {
  final (String, String)? region;
  final bool alarmDefault;
  final Future<void> Function() onDone;
  const _PickupSetupStep({
    required this.region,
    required this.alarmDefault,
    required this.onDone,
  });

  @override
  State<_PickupSetupStep> createState() => _PickupSetupStepState();
}

class _PickupSetupStepState extends State<_PickupSetupStep> {
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
                    _StepBadge('3/3'),
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
            child: _PrimaryButton(
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

class _ApartmentFinishStep extends StatefulWidget {
  final (String, String)? region;
  final Future<void> Function() onDone;
  const _ApartmentFinishStep({required this.region, required this.onDone});

  @override
  State<_ApartmentFinishStep> createState() => _ApartmentFinishStepState();
}

class _ApartmentFinishStepState extends State<_ApartmentFinishStep> {
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
                    _StepBadge('3/3'),
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
            child: _PrimaryButton(
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
