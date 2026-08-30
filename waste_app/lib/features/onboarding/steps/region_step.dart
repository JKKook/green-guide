/// 온보딩 ② 지역 선택 (시/도 → 시/군/구 시트).
library;

import 'package:flutter/material.dart';

import '../../../data/haptics.dart';
import '../../../data/region_data.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import '../../../widgets/korea_map.dart';
import '../../../widgets/region_picker.dart' show locateRegionFromGps;
import '../widgets/onboarding_primitives.dart';

class RegionStep extends StatefulWidget {
  final ValueChanged<(String, String)?> onDone;
  const RegionStep({super.key, required this.onDone});

  @override
  State<RegionStep> createState() => _RegionStepState();
}


class _RegionStepState extends State<RegionStep> {
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
      builder: (_) => SigunguSheet(sido: sido),
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
                    const StepBadge('1/3'),
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
                OnboardingButton(
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
                OnboardingButton(
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
class SigunguSheet extends StatefulWidget {
  final String sido;
  const SigunguSheet({super.key, required this.sido});

  @override
  State<SigunguSheet> createState() => _SigunguSheetState();
}


class _SigunguSheetState extends State<SigunguSheet> {
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
            OnboardingButton(
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
