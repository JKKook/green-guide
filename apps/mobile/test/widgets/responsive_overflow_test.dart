/// 반응형 텍스트 레이아웃 전수 점검 — 좁은 폭(320~412dp) × 큰 글꼴 배율(1.0~2.0)
/// 조합으로 주요 화면을 그려 RenderFlex overflow 를 수집한다.
/// 실제 본문 폰트(Pretendard)를 로드해 글자 폭이 실기기와 같게 측정한다.
library;

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/data/legal_terms.dart';
import 'package:greenguide/features/capture/capture_entry_sheet.dart';
import 'package:greenguide/features/history/history_screen.dart';
import 'package:greenguide/features/history/widgets/range_sheet.dart';
import 'package:greenguide/features/home/home_screen.dart';
import 'package:greenguide/features/home/widgets/how_sheet.dart';
import 'package:greenguide/features/onboarding/housing_type_sheet.dart';
import 'package:greenguide/features/onboarding/onboarding_screen.dart';
import 'package:greenguide/features/onboarding/steps/apartment_finish_step.dart';
import 'package:greenguide/features/onboarding/steps/pickup_setup_step.dart';
import 'package:greenguide/features/onboarding/steps/region_step.dart';
import 'package:greenguide/features/result/result_modal.dart';
import 'package:greenguide/features/result/widgets/feedback_sheet.dart';
import 'package:greenguide/features/schedule/collection_reminders_screen.dart';
import 'package:greenguide/features/schedule/collection_schedule_screen.dart';
import 'package:greenguide/features/schedule/pickup_weekdays_sheet.dart';
import 'package:greenguide/features/schedule/reminder_sheet.dart';
import 'package:greenguide/features/search/unified_search_screen.dart';
import 'package:greenguide/features/settings/settings_screen.dart';
import 'package:greenguide/features/settings/terms_screen.dart';
import 'package:greenguide/features/shell/main_shell.dart';
import 'package:greenguide/theme/app_theme.dart';
import 'package:image/image.dart' as img;

import '../helpers/result_fakes.dart';
import '../helpers/test_env.dart';

/// 화면 폭·높이(dp) × 글꼴 배율. 폭 320(iPhone SE 1세대·소형 안드로이드) 부터
/// 412(일반 안드로이드) 까지, 배율은 OS 접근성 '크게' 범위(1.3~2.0).
const _configs = <(double, double, double)>[
  (320, 568, 1.0),
  (320, 568, 1.3),
  (360, 640, 1.3),
  (360, 640, 1.5),
  (360, 780, 2.0),
  (390, 844, 1.5),
  (412, 915, 2.0),
];

typedef _Scenario = ({
  String name,
  Widget Function(BuildContext) home,
  Future<void> Function(WidgetTester)? after,
});

/// 시트형 화면은 버튼을 눌러 열어야 하므로 열기 버튼이 있는 Scaffold 를 홈으로 둔다.
Widget _opener(void Function(BuildContext) open) => Builder(
  builder: (context) => Scaffold(
    body: Center(
      child: TextButton(
        onPressed: () => open(context),
        child: const Text('open'),
      ),
    ),
  ),
);

Future<void> _tapOpen(WidgetTester tester) async {
  await tester.tap(find.text('open'));
  await tester.pump();
  await tester.pump(const Duration(seconds: 1));
}

/// 한 패밀리의 굵기 파일은 FontLoader 하나에 모두 넣고 한 번만 load 한다
/// (파일마다 따로 load 하면 적용되지 않고 테스트 기본 글꼴로 측정된다).
Future<void> _loadFonts() async {
  const families = {
    kBodyFontFamily: [
      'Pretendard-Regular.otf',
      'Pretendard-Medium.otf',
      'Pretendard-SemiBold.otf',
      'Pretendard-Bold.otf',
    ],
    'PureunSup': ['PureunSup-Bold.otf'],
  };
  for (final MapEntry(key: family, value: files) in families.entries) {
    final loader = FontLoader(family);
    for (final file in files) {
      final bytes = File('assets/fonts/$file').readAsBytesSync();
      loader.addFont(Future.value(ByteData.sublistView(bytes)));
    }
    await loader.load();
  }
}

/// overflow 를 낸 위젯의 생성 위치(`lib/....dart:줄:칸`) — 없으면 위젯 체인.
String _locationOf(FlutterErrorDetails details) {
  final props = details.informationCollector?.call() ?? const [];
  final transformed = debugTransformDebugCreator(props);
  final text = transformed.map((n) => n.toStringDeep()).join('\n');
  final m = RegExp(r'file:///\S+?\.dart:\d+:\d+').firstMatch(text);
  if (m != null) return m.group(0)!.replaceFirst(RegExp(r'.*/lib/'), 'lib/');
  for (final p in props) {
    final v = p.value;
    if (v is DebugCreator) return v.element.debugGetCreatorChain(6);
  }
  return '?';
}

void main() {
  late Directory dir;
  setUpAll(() async {
    dir = await setUpTestEnv(
      prefs: {
        'onboarding_done': true,
        'region_prompt_shown': true,
        'region_sido': '서울특별시',
        'region_sigungu': '강남구',
        'housing_type': 'house',
        'pickup_weekdays': '2,5',
        'collection_alarm_opt_in': true,
      },
    );
    await _loadFonts();
  });
  tearDownAll(() => dir.delete(recursive: true));

  final scenarios = <_Scenario>[
    (name: '온보딩 동의', home: (_) => const OnboardingScreen(), after: null),
    (
      name: '온보딩 지역',
      home: (_) => Scaffold(body: RegionStep(onDone: (_) {})),
      after: null,
    ),
    (
      name: '온보딩 시군구 시트',
      home: (_) => _opener(
        (c) => showModalBottomSheet<String>(
          context: c,
          isScrollControlled: true,
          builder: (_) => const SigunguSheet(sido: '경기도'),
        ),
      ),
      after: _tapOpen,
    ),
    (
      name: '온보딩 주거 시트',
      home: (_) => _opener((c) => showHousingTypeSheet(c, stepLabel: '2/3')),
      after: _tapOpen,
    ),
    (
      name: '온보딩 수거 설정',
      home: (_) => Scaffold(
        body: PickupSetupStep(
          region: ('서울특별시', '강남구'),
          alarmDefault: true,
          onDone: () async {},
        ),
      ),
      after: null,
    ),
    (
      name: '온보딩 아파트 마무리',
      home: (_) => Scaffold(
        body: ApartmentFinishStep(
          region: ('서울특별시', '강남구'),
          onDone: () async {},
        ),
      ),
      after: null,
    ),
    (name: '메인 셸', home: (_) => const MainShell(), after: null),
    (name: '홈', home: (_) => const HomeScreen(), after: null),
    (
      name: '홈 사용법 시트',
      home: (_) => _opener(
        (c) => showModalBottomSheet<void>(
          context: c,
          showDragHandle: true,
          builder: (_) => const HowSheet(),
        ),
      ),
      after: _tapOpen,
    ),
    (name: '검색', home: (_) => const UnifiedSearchScreen(), after: null),
    (name: '기록', home: (_) => const HistoryScreen(), after: null),
    (name: '설정', home: (_) => const SettingsScreen(), after: null),
    (name: '약관 목록', home: (_) => const TermsListScreen(), after: null),
    (
      name: '약관 상세',
      home: (_) => TermsDetailScreen(doc: kLegalDocs.first),
      after: null,
    ),
    (
      name: '수거 일정',
      home: (_) => const CollectionScheduleScreen(),
      after: null,
    ),
    (
      name: '수거 알림',
      home: (_) => const CollectionRemindersScreen(),
      after: null,
    ),
    (
      name: '결과 모달(스마트촬영·표시 판정)',
      home: (_) => _opener((c) {
        final image = File('${dir.path}/shot2.png')
          ..writeAsBytesSync(img.encodePng(img.Image(width: 64, height: 48)));
        showResultModal(
          c,
          image,
          isSmartCapture: true,
          prediction: FakePredictionService(markPriorityJson()),
          api: () async => FakeApi(),
        );
      }),
      after: (tester) async {
        await _tapOpen(tester);
        await settleIo(tester, const Duration(seconds: 1));
        await tester.pump(const Duration(seconds: 2));
      },
    ),
    (
      name: '피드백 시트',
      home: (_) => _opener(
        (c) => showModalBottomSheet<void>(
          context: c,
          isScrollControlled: true,
          builder: (_) => FeedbackSheet(
            prediction: Prediction.fromHierJson(markPriorityJson()),
          ),
        ),
      ),
      after: _tapOpen,
    ),
    (
      name: '촬영 진입 시트',
      home: (_) => _opener(showCaptureEntrySheet),
      after: _tapOpen,
    ),
    (
      name: '기록 기간 시트',
      home: (_) => _opener(
        (c) => showModalBottomSheet<void>(
          context: c,
          isScrollControlled: true,
          builder: (_) => const RangeSheet(),
        ),
      ),
      after: _tapOpen,
    ),
    (
      name: '알림 시트',
      home: (_) => _opener(
        (c) => showReminderSheet(c, weekday: 2, allowWeekdayPick: true),
      ),
      after: _tapOpen,
    ),
    (
      name: '수거 요일 시트',
      home: (_) => _opener((c) => showPickupWeekdaysSheet(c, current: [2, 5])),
      after: _tapOpen,
    ),
    (
      name: '앱 정보 다이얼로그',
      home: (_) => const SettingsScreen(),
      after: (tester) async {
        await tester.scrollUntilVisible(find.text('앱 정보'), 200,
            scrollable: find.byType(Scrollable).first);
        await tester.tap(find.text('앱 정보'));
        await tester.pump();
        await tester.pump(const Duration(seconds: 1));
      },
    ),
    (
      name: '결과 모달(오류)',
      home: (_) => _opener((c) {
        final image = File('${dir.path}/shot.png')
          ..writeAsBytesSync(img.encodePng(img.Image(width: 8, height: 8)));
        showResultModal(c, image);
      }),
      after: (tester) async {
        await _tapOpen(tester);
        await settleIo(tester, const Duration(seconds: 1));
        await tester.pump(const Duration(seconds: 2));
      },
    ),
  ];

  for (final s in scenarios) {
    testWidgets('${s.name} — overflow 없음', (tester) async {
      final overflows = <String>[];
      final themes = [('light', buildLightTheme()), ('dark', buildDarkTheme())];
      for (final ((w, h, scale), (mode, theme)) in [
        for (final c in _configs)
          for (final t in themes) (c, t),
      ]) {
        tester.view.physicalSize = Size(w, h);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.reset);

        final prev = FlutterError.onError;
        FlutterError.onError = (details) {
          final text = details.toString();
          if (text.contains('overflowed')) {
            final loc = _locationOf(details);
            overflows.add(
              '${w.toInt()}x${h.toInt()}@$scale/$mode ${details.exceptionAsString()} ← $loc',
            );
          } else {
            prev?.call(details);
          }
        };
        try {
          await tester.pumpWidget(
            MaterialApp(
              theme: theme,
              locale: const Locale('ko'),
              builder: (context, child) => MediaQuery(
                data: MediaQuery.of(context).copyWith(
                  textScaler: TextScaler.linear(scale),
                ),
                child: child!,
              ),
              home: Builder(builder: s.home),
            ),
          );
          await settleIo(tester);
          await tester.pump(const Duration(seconds: 1));
          await s.after?.call(tester);
          // 다음 조합 전에 트리를 비워 상태를 초기화한다.
          await tester.pumpWidget(const SizedBox());
        } finally {
          FlutterError.onError = prev;
        }
      }
      expect(overflows, isEmpty, reason: overflows.join('\n'));
    });
  }
}
