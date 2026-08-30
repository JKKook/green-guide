import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'data/haptics.dart';
import 'data/licenses.dart';
import 'data/settings_store.dart';
import 'screens/splash_screen.dart';
import 'services/class_loader.dart';
import 'services/server_warmup.dart';
import 'theme/app_theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  registerBundledLicenses();
  await initAppThemeMode();
  await initAppHousingType();
  await initHaptics();
  // 절전 중인 서버 깨우기 — 첫 분석이 타임아웃으로 실패하지 않도록 미리 핑.
  unawaited(ServerWarmup.ping());
  // 클래스 레지스트리 비동기 fetch (실패해도 앱 동작)
  unawaited(ClassLoader().loadFromServer());
  runApp(const GreenGuideApp());
}

// `unawaited` 명시 import 회피
void unawaited(Future<void> future) {}

class GreenGuideApp extends StatelessWidget {
  const GreenGuideApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: appThemeMode,
      builder: (_, mode, _) => MaterialApp(
        title: '그린가이드',
        theme: buildLightTheme(),
        darkTheme: buildDarkTheme(),
        themeMode: mode,
        // 한글 로컬라이제이션 — DatePicker·다이얼로그 등 시스템 위젯
        locale: const Locale('ko'),
        supportedLocales: const [Locale('ko'), Locale('en')],
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        debugShowCheckedModeBanner: false,
        home: const SplashRouter(),
      ),
    );
  }
}
