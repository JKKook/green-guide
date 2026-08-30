import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'data/settings_store.dart';
import 'screens/splash_screen.dart';
import 'theme/app_theme.dart';

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
