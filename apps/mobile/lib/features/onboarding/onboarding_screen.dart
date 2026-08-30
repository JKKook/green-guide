import 'package:flutter/material.dart';

import '../../core/di/app_scope.dart';
import '../../data/haptics.dart';
import '../../data/settings_store.dart';
import '../shell/main_shell.dart';
import 'housing_type_sheet.dart';
import 'steps/apartment_finish_step.dart';
import 'steps/consent_step.dart';
import 'steps/pickup_setup_step.dart';
import 'steps/region_step.dart';

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
      0 => ConsentStep(onDone: _onConsented),
      1 => RegionStep(onDone: _onRegionDone),
      _ => _housing == HousingType.house
          ? PickupSetupStep(
              region: _region,
              alarmDefault: _alarmOptIn,
              onDone: _finish,
            )
          : ApartmentFinishStep(region: _region, onDone: _finish),
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
