import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';

import '../api/api_client.dart';
import '../core/di/app_scope.dart';
import '../core/feedback/app_snackbar.dart';
import '../core/ui/ds_card.dart';
import '../data/collection_schedule.dart';
import '../data/haptics.dart';
import '../data/settings_store.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/region_picker.dart';
import 'collection_schedule_screen.dart';
import 'onboarding_screen.dart';
import 'splash_screen.dart';
import 'terms_screen.dart';

/// 설정 — 시안 5a: 내 동네 / 알림 / 분류 / 화면 / 정보.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final SettingsStore _store = AppScope.settings;
  final ReminderStore _reminders = ReminderStore();
  final TextEditingController _urlController = TextEditingController();
  bool _loading = true;
  bool _testing = false;
  String? _testResult;
  bool _testSuccess = false;
  bool _hapticsEnabled = true;
  bool _tipsNotification = false;
  int _reminderCount = 0;
  (String, String)? _region;
  HousingType? _housing;
  List<int> _pickupDays = const [];
  bool _devMode = false; // API 서버 섹션 노출 (버전 7탭)
  int _versionTaps = 0;
  String _version = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final url = await _store.getApiUrl();
    final haptics = await _store.isHapticsEnabled();
    final tips = await _store.isTipsNotificationEnabled();
    final region = await _store.getRegion();
    final reminders = await _reminders.load();
    final housing = await _store.getHousingType();
    final pickupDays = await _store.getPickupWeekdays();
    final dev = await _store.isDevOptionsEnabled();
    String version = '';
    try {
      final info = await PackageInfo.fromPlatform();
      version = '${info.version} (${info.buildNumber})';
    } catch (_) {}
    if (!mounted) return;
    setState(() {
      _devMode = dev;
      _version = version;
      _urlController.text = url;
      _hapticsEnabled = haptics;
      _tipsNotification = tips;
      _reminderCount = reminders.length;
      _region = region;
      _housing = housing;
      _pickupDays = pickupDays;
      _loading = false;
    });
  }

  Future<void> _save() async {
    Haptics.selection();
    final url = _urlController.text.trim();
    if (url.isEmpty) return;
    await _store.setApiUrl(url);
    if (!mounted) return;
    showAppSnackBar(context, '저장됨');
  }

  Future<void> _testConnection() async {
    Haptics.selection();
    setState(() {
      _testing = true;
      _testResult = null;
    });
    final url = _urlController.text.trim();
    final client = WasteApiClient(baseUrl: url);
    final ok = await client.isHealthy();
    // 성공 시 즉시 저장 — '연결 성공' 을 보고도 저장을 안 눌러 실제 분류는
    // 이전 서버로 나가던 함정 제거 (테스트 성공 = 이 서버 사용 의사로 간주)
    if (ok && url.isNotEmpty) await _store.setApiUrl(url);
    if (!mounted) return;
    setState(() {
      _testing = false;
      _testSuccess = ok;
      _testResult =
          ok ? '연결 성공 — 이 서버로 저장됨' : '연결 실패 — URL 또는 서버 상태를 확인하세요';
    });
  }

  void _useTestMode() {
    Haptics.selection();
    setState(() {
      _urlController.text = SettingsStore.testModeApiUrl;
      _testResult = null;
    });
  }

  void _useDefault() {
    Haptics.selection();
    setState(() {
      _urlController.text = SettingsStore.defaultApiUrl;
      _testResult = null;
    });
  }

  Future<void> _showAbout() async {
    Haptics.selection();
    final info = await PackageInfo.fromPlatform();
    if (!mounted) return;
    showAboutDialog(
      context: context,
      applicationName: '그린가이드',
      applicationVersion: '${info.version} (build ${info.buildNumber})',
      applicationIcon: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.asset('assets/icon/icon.png', width: 48, height: 48),
      ),
      applicationLegalese:
          '© 2026 GreenGuide AI\n\n'
          '딥러닝 기반 폐기물 분류 보조 도구 — 대분류·세부품목 계층 분류 모델.\n\n'
          '본 서비스의 재질 분류 모델은 과학기술정보통신부와 '
          '한국지능정보사회진흥원(NIA)이 지원하는 AI 통합 플랫폼 AI-Hub '
          '(aihub.or.kr)의 학습용 데이터를 활용하여 학습되었습니다. '
          'TACO·Open Images 등 공개 데이터셋을 함께 활용했습니다.\n\n'
          '지역별 배출 정보 출처: 행정안전부 전국생활쓰레기배출정보표준데이터 '
          '(공공데이터포털).\n\n'
          '자세한 고지는 아래 "오픈소스 라이선스" 에서 확인할 수 있어요.\n\n'
          '※ 결과는 참고용이며, 실제 분리수거는 거주 지자체 가이드를 우선 따르세요.',
    );
  }

  /// 버전 7회 탭 → 개발자 옵션(API 서버) 토글 — 일반 사용자에게는 숨김.
  Future<void> _onVersionTap() async {
    _versionTaps++;
    if (_versionTaps < 7) {
      // 힌트는 디버그 빌드에서만 — 베타 사용자에게 개발자 옵션을 광고하지 않는다.
      if (_versionTaps >= 4 && kDebugMode) {
        showAppSnackBar(
          context,
          _devMode
              ? '${7 - _versionTaps}번 더 누르면 개발자 옵션을 숨겨요'
              : '${7 - _versionTaps}번 더 누르면 개발자 옵션이 열려요',
          duration: const Duration(milliseconds: 900),
          replace: true,
        );
      }
      return;
    }
    _versionTaps = 0;
    Haptics.medium();
    final next = !_devMode;
    await _store.setDevOptionsEnabled(next);
    if (!mounted) return;
    setState(() => _devMode = next);
    showAppSnackBar(
      context,
      next ? '개발자 옵션이 열렸어요' : '개발자 옵션을 숨겼어요',
      replace: true,
    );
  }

  /// 개발용 — 더미 기록 10건 (배너 이미지를 사진으로, 최근 2주에 분산).
  Future<void> _seedDummyHistory() async {
    Haptics.selection();
    const samples = [
      ('styrofoam_white', 0.92, 'assets/banners/tip_13a.png', 0, 14, 32),
      ('carton', 0.88, 'assets/banners/tip_13b.png', 0, 9, 10),
      ('pet', 0.95, 'assets/banners/tip_13c.png', 1, 18, 2),
      ('trash_other', 0.71, 'assets/banners/tip_13d.png', 2, 20, 45),
      ('glass_clear', 0.83, 'assets/banners/tip_13e.png', 3, 11, 5),
      ('vinyl_clean', 0.79, 'assets/banners/tip_13f.png', 5, 16, 40),
      ('paper', 0.90, 'assets/banners/tip_13g.png', 6, 8, 25),
      ('metal', 0.86, 'assets/banners/tip_13h.png', 8, 19, 12),
      ('battery', 0.77, 'assets/banners/tip_13i.png', 10, 13, 50),
      ('clothes', 0.81, 'assets/banners/tip_13j.png', 13, 17, 30),
    ];
    final repo = AppScope.history;
    final tmp = await getTemporaryDirectory();
    final now = DateTime.now();
    var n = 0;
    String? lastError;
    for (final (cls, conf, asset, daysAgo, hour, minute) in samples) {
      try {
        final bytes = await rootBundle.load(asset);
        final f = File('${tmp.path}/dummy_${n}_${asset.split('/').last}');
        await f.writeAsBytes(bytes.buffer.asUint8List());
        final day = now.subtract(Duration(days: daysAgo));
        await repo.save(
          sourceImage: f,
          predictedClass: cls,
          confidence: conf,
          uploadId: null,
          modelArch: 'dummy-seed',
          createdAt: DateTime(day.year, day.month, day.day, hour, minute),
        );
        n++;
      } catch (e) {
        lastError = '$e';
      }
    }
    if (!mounted) return;
    showAppSnackBar(
      context,
      lastError == null
          ? '더미 기록 $n건을 추가했어요 — 기록 탭에서 당겨서 새로고침'
          : '더미 기록 $n건 추가 · 실패: $lastError',
      duration: const Duration(seconds: 5),
    );
  }

  Future<void> _clearHistory() async {
    Haptics.selection();
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('기록 전체 삭제'),
        content: const Text('더미를 포함한 모든 분류 기록을 지웁니다. 되돌릴 수 없어요.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('취소')),
          FilledButton.tonal(
            style: FilledButton.styleFrom(
                minimumSize: const Size(0, 44),
                padding: const EdgeInsets.symmetric(horizontal: kSpaceL)),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('삭제'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await AppScope.history.clear();
    if (!mounted) return;
    showAppSnackBar(context, '기록을 모두 지웠어요');
  }

  /// 다크 모드 스위치 — 명시적 라이트/다크 전환.
  Future<void> _setDark(bool dark) async {
    Haptics.selection();
    final mode = dark ? ThemeMode.dark : ThemeMode.light;
    await _store.setThemeMode(mode);
    appThemeMode.value = mode;
  }

  /// 다크 모드 행 탭 — 시스템 → 라이트 → 다크 순환.
  Future<void> _cycleThemeMode() async {
    Haptics.selection();
    final next = switch (appThemeMode.value) {
      ThemeMode.system => ThemeMode.light,
      ThemeMode.light => ThemeMode.dark,
      ThemeMode.dark => ThemeMode.system,
    };
    await _store.setThemeMode(next);
    appThemeMode.value = next;
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : ListView(
                padding: EdgeInsets.fromLTRB(
                  20,
                  kSpaceM,
                  20,
                  kSpaceXXL + 16 + MediaQuery.viewPaddingOf(context).bottom,
                ),
                children: [
                  const Padding(
                    padding: EdgeInsets.only(bottom: 16),
                    child: Text(
                      '설정',
                      style:
                          TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                  ),

                  _SectionLabel(tokens: t, label: '내 동네'),
                  DsCard(
                    elevated: true,
                    clipBehavior: Clip.antiAlias,
                    child: Column(
                      children: [
                        _DsRow(
                          tokens: t,
                          icon: Icons.place_outlined,
                          title: _region?.$2 ?? '지역 미설정',
                          subtitle: _region == null
                              ? '지역별 배출 기준 안내에 사용해요'
                              : '${_region!.$2} 기준으로 배출 정보를 안내해요',
                          trailing: Text(
                            '변경',
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: t.accentChipText,
                            ),
                          ),
                          onTap: () async {
                            final picked = await showRegionPicker(context);
                            if (picked != null && mounted) {
                              setState(() => _region = picked);
                            }
                          },
                        ),
                        Container(height: 1, color: t.border),
                        // 주거 형태 — 온보딩 ③ 세대 구분 (언제든 변경)
                        _DsRow(
                          tokens: t,
                          icon: _housing == HousingType.apartment
                              ? Icons.apartment_outlined
                              : Icons.home_outlined,
                          title: switch (_housing) {
                            HousingType.apartment => '아파트 · 오피스텔',
                            HousingType.house => '주택 · 빌라',
                            null => '주거 형태 미설정',
                          },
                          subtitle: switch (_housing) {
                            HousingType.apartment => '단지 내 분리배출장에 상시 배출',
                            HousingType.house => '동네 수거 요일에 맞춰 문 앞 배출',
                            null => '주거 형태에 따라 안내가 달라져요',
                          },
                          trailing: Text(
                            '변경',
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: t.accentChipText,
                            ),
                          ),
                          onTap: () async {
                            final picked = await showHousingTypeSheet(context,
                                current: _housing);
                            if (picked != null && mounted) {
                              await _store.setHousingType(picked);
                              setState(() => _housing = picked);
                            }
                          },
                        ),
                        // 수거 요일 — 주택·빌라만 (아파트는 상시 배출)
                        if (_housing != HousingType.apartment) ...[
                          Container(height: 1, color: t.border),
                          _DsRow(
                            tokens: t,
                            icon: Icons.event_repeat_outlined,
                            title: '우리 집 수거 요일',
                            subtitle: _pickupDays.isEmpty
                                ? '동네 기본값 사용 · 탭해서 직접 지정'
                                : '매주 ${_pickupDays.map((d) => kDayNames[d - 1]).join(' · ')} · 재활용품 문 앞 배출',
                            trailing: Text(
                              '변경',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w700,
                                color: t.accentChipText,
                              ),
                            ),
                            onTap: () async {
                              final picked = await showPickupWeekdaysSheet(
                                  context,
                                  current: _pickupDays);
                              if (picked != null && mounted) {
                                await _store.setPickupWeekdays(picked);
                                setState(() => _pickupDays = picked);
                              }
                            },
                          ),
                        ],
                      ],
                    ),
                  ),

                  _SectionLabel(tokens: t, label: '알림'),
                  DsCard(
                    elevated: true,
                    clipBehavior: Clip.antiAlias,
                    child: Column(
                      children: [
                        _DsRow(
                          tokens: t,
                          icon: Icons.notifications_none,
                          title: '수거일 알림',
                          subtitle: _reminderCount == 0
                              ? '등록된 알림 없음 · $kReminderPendingNote'
                              : '알림 $_reminderCount개 · $kReminderPendingNote',
                          trailing: Icon(Icons.chevron_right,
                              size: 16, color: t.faint),
                          onTap: () async {
                            Haptics.selection();
                            await Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) =>
                                    const CollectionRemindersScreen(),
                              ),
                            );
                            _load();
                          },
                        ),
                        Container(height: 1, color: t.border),
                        _DsRow(
                          tokens: t,
                          icon: Icons.tips_and_updates_outlined,
                          title: '오늘의 팁 알림',
                          subtitle: '하루 한 번, 분리배출 팁 · $kReminderPendingNote',
                          trailing: Switch(
                            value: _tipsNotification,
                            activeTrackColor: brandSeed,
                            onChanged: (v) async {
                              Haptics.selection();
                              await _store.setTipsNotificationEnabled(v);
                              setState(() => _tipsNotification = v);
                            },
                          ),
                        ),
                      ],
                    ),
                  ),

                  _SectionLabel(tokens: t, label: '분류'),
                  DsCard(
                    elevated: true,
                    clipBehavior: Clip.antiAlias,
                    child: _DsRow(
                      tokens: t,
                      icon: Icons.vibration,
                      title: '햅틱 피드백',
                      subtitle: '버튼·결과 표시 시 진동',
                      trailing: Switch(
                        value: _hapticsEnabled,
                        activeTrackColor: brandSeed,
                        onChanged: (v) async {
                          Haptics.selection();
                          await _store.setHapticsEnabled(v);
                          setHapticsEnabled(v); // 전역 즉시 반영
                          setState(() => _hapticsEnabled = v);
                        },
                      ),
                    ),
                  ),

                  _SectionLabel(tokens: t, label: '화면'),
                  ValueListenableBuilder<ThemeMode>(
                    valueListenable: appThemeMode,
                    builder: (context, mode, _) => DsCard(
                      elevated: true,
                      clipBehavior: Clip.antiAlias,
                      child: _DsRow(
                        tokens: t,
                        icon: Icons.dark_mode_outlined,
                        title: '다크 모드',
                        subtitle: switch (mode) {
                          ThemeMode.system => '시스템 설정 따르기 (탭해서 변경)',
                          ThemeMode.light => '라이트 고정 (탭해서 변경)',
                          ThemeMode.dark => '다크 고정 (탭해서 변경)',
                        },
                        trailing: Switch(
                          value: t.dark,
                          activeTrackColor: brandSeed,
                          onChanged: _setDark,
                        ),
                        onTap: _cycleThemeMode,
                      ),
                    ),
                  ),

                  _SectionLabel(tokens: t, label: '정보'),
                  DsCard(
                    elevated: true,
                    clipBehavior: Clip.antiAlias,
                    child: Column(
                      children: [
                        _DsRow(
                          tokens: t,
                          icon: Icons.info_outline,
                          title: '앱 정보',
                          subtitle: '버전·라이선스·크레딧',
                          trailing: Icon(Icons.chevron_right,
                              size: 16, color: t.faint),
                          onTap: _showAbout,
                        ),
                        Container(height: 1, color: t.border),
                        _DsRow(
                          tokens: t,
                          icon: Icons.description_outlined,
                          title: '약관 및 정책',
                          subtitle: '이용약관 · 개인정보 · 위치기반 서비스 · 선택 동의',
                          trailing: Icon(Icons.chevron_right,
                              size: 16, color: t.faint),
                          onTap: () {
                            Haptics.selection();
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                  builder: (_) => const TermsListScreen()),
                            );
                          },
                        ),
                        Container(height: 1, color: t.border),
                        _DsRow(
                          tokens: t,
                          icon: Icons.verified_outlined,
                          title: '버전',
                          subtitle: _version.isEmpty ? '-' : _version,
                          onTap: _onVersionTap,
                        ),
                      ],
                    ),
                  ),

                  if (_devMode) ...[
                    _SectionLabel(tokens: t, label: '개발자'),
                    DsCard(
                      elevated: true,
                      clipBehavior: Clip.antiAlias,
                      child: _DsRow(
                        tokens: t,
                        icon: Icons.replay_outlined,
                        title: '온보딩 다시 보기',
                        subtitle: '첫 실행 흐름(동의 → 지역 → 세대 구분)을 다시 실행',
                        trailing: Icon(Icons.chevron_right,
                            size: 16, color: t.faint),
                        onTap: () async {
                          Haptics.selection();
                          await _store.resetOnboarding();
                          if (!context.mounted) return;
                          Navigator.of(context).pushAndRemoveUntil(
                            MaterialPageRoute(
                                builder: (_) => const SplashRouter()),
                            (_) => false,
                          );
                        },
                      ),
                    ),
                    DsCard(
                      elevated: true,
                      clipBehavior: Clip.antiAlias,
                      margin: const EdgeInsets.only(top: 8),
                      child: Column(
                        children: [
                          _DsRow(
                            tokens: t,
                            icon: Icons.auto_awesome_motion_outlined,
                            title: '더미 기록 10건 추가',
                            subtitle: 'UI 확인용 — 최근 2주에 분산된 샘플 기록',
                            trailing: Icon(Icons.chevron_right,
                                size: 16, color: t.faint),
                            onTap: _seedDummyHistory,
                          ),
                          Container(height: 1, color: t.border),
                          _DsRow(
                            tokens: t,
                            icon: Icons.delete_sweep_outlined,
                            title: '기록 전체 삭제',
                            subtitle: '더미 포함 모든 분류 기록 제거',
                            trailing: Icon(Icons.chevron_right,
                                size: 16, color: t.faint),
                            onTap: _clearHistory,
                          ),
                        ],
                      ),
                    ),
                    _SectionLabel(tokens: t, label: '개발자 — API 서버'),
                    DsCard(
                      elevated: true,
                      clipBehavior: Clip.antiAlias,
                      child: Padding(
                        padding: const EdgeInsets.all(kSpaceL),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            TextField(
                              controller: _urlController,
                              decoration: InputDecoration(
                                labelText: 'API URL',
                                hintText: 'http://10.0.2.2:8000',
                                prefixIcon: Icon(
                                  Icons.link,
                                  color: cs.onSurfaceVariant,
                                ),
                              ),
                              keyboardType: TextInputType.url,
                              autocorrect: false,
                            ),
                            const SizedBox(height: kSpaceS),
                            Wrap(
                              spacing: kSpaceS,
                              children: [
                                ActionChip(
                                  avatar: const Icon(Icons.cloud, size: 16),
                                  label: const Text('프로덕션 (HF Spaces)'),
                                  onPressed: _useDefault,
                                ),
                                ActionChip(
                                  avatar: const Icon(Icons.usb, size: 16),
                                  label: const Text('로컬 (localhost)'),
                                  onPressed: _useTestMode,
                                ),
                              ],
                            ),
                            const SizedBox(height: kSpaceL),
                            Row(
                              children: [
                                Expanded(
                                  child: OutlinedButton.icon(
                                    onPressed:
                                        _testing ? null : _testConnection,
                                    icon: _testing
                                        ? const SizedBox(
                                            width: 16,
                                            height: 16,
                                            child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                            ),
                                          )
                                        : const Icon(Icons.wifi_tethering),
                                    label: const Text('연결 테스트'),
                                  ),
                                ),
                                const SizedBox(width: kSpaceM),
                                Expanded(
                                  child: FilledButton.icon(
                                    onPressed: _save,
                                    icon: const Icon(Icons.save_outlined),
                                    label: const Text('저장'),
                                  ),
                                ),
                              ],
                            ),
                            if (_testResult != null) ...[
                              const SizedBox(height: kSpaceM),
                              Container(
                                padding: const EdgeInsets.all(kSpaceM),
                                decoration: BoxDecoration(
                                  color: _testSuccess
                                      ? cs.primaryContainer
                                      : cs.errorContainer,
                                  borderRadius:
                                      BorderRadius.circular(kRadiusMedium),
                                ),
                                child: Row(
                                  children: [
                                    Icon(
                                      _testSuccess
                                          ? Icons.check_circle
                                          : Icons.error,
                                      color: _testSuccess
                                          ? cs.onPrimaryContainer
                                          : cs.onErrorContainer,
                                    ),
                                    const SizedBox(width: kSpaceS),
                                    Expanded(
                                      child: Text(
                                        _testResult!,
                                        style: TextStyle(
                                          color: _testSuccess
                                              ? cs.onPrimaryContainer
                                              : cs.onErrorContainer,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ],
                ],
              ),
      ),
    );
  }
}

/// 섹션 라벨 — 시안: 11px w700 accent-800.
class _SectionLabel extends StatelessWidget {
  final DsTokens tokens;
  final String label;
  const _SectionLabel({required this.tokens, required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 20, bottom: 8),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.22,
          color: tokens.accentChipText,
        ),
      ),
    );
  }
}

/// 설정 행 — 아이콘(accent-700) + 제목/부제 + 트레일링.
class _DsRow extends StatelessWidget {
  final DsTokens tokens;
  final IconData icon;
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final VoidCallback? onTap;
  const _DsRow({
    required this.tokens,
    required this.icon,
    required this.title,
    this.subtitle,
    this.trailing,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final row = Padding(
      padding: const EdgeInsets.fromLTRB(15, 14, 15, 14),
      child: Row(
        children: [
          Icon(icon, size: 20, color: tokens.accentStrong),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                      fontSize: 14, fontWeight: FontWeight.w700),
                ),
                if (subtitle != null) ...[
                  const SizedBox(height: 2),
                  Text(
                    subtitle!,
                    style: TextStyle(fontSize: 11, color: tokens.muted2),
                  ),
                ],
              ],
            ),
          ),
          ?trailing,
        ],
      ),
    );
    if (onTap == null) return row;
    return InkWell(onTap: onTap, child: row);
  }
}
