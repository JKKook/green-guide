import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// GreenGuide 시드 (브랜드용). primary 만 추출하고
/// secondary/tertiary 는 별도 시드로 다양한 팔레트 확보.
/// 2026-08 홈 리뉴얼 시안 기준 — 스틸블루 모노크롬 팔레트.
const Color brandSeed = Color(0xFF5980A6);      // Steel Blue — 시안 accent
const Color secondarySeed = Color(0xFF94BCE3);  // Light Blue — 보조 강조
const Color tertiarySeed = Color(0xFF728FAB);   // Gray Blue — 포인트 (accent-2)

/// 디스플레이(제목·카피) 전용 폰트 — 유한킴벌리 푸른숲체 (YK Green Forest).
/// Light 300 · Medium 400(normal) · Bold 700. 이전 폰트는 'Jua' (단일 웨이트).
const String kDisplayFontFamily = 'PureunSup';

/// 본문 기본 폰트 — 시안(Claude Design 'GreenGuide Design') 확정값.
const String kBodyFontFamily = 'Pretendard';

/// Industry 디자인 시스템 톤 램프 — 시안 styles.css 실측값.
/// OKLCH 공유 명도 스케일이라 M3 파생색과 별개로 시안 UI 에 직접 쓴다.
const Color kNeutral100 = Color(0xFFF5F5F8);
const Color kNeutral200 = Color(0xFFE7E7EA);
const Color kNeutral300 = Color(0xFFD4D4D7);
const Color kNeutral400 = Color(0xFFB7B7BA);
const Color kNeutral500 = Color(0xFF98989B);
const Color kNeutral600 = Color(0xFF7A7A7D);
const Color kAccent100 = Color(0xFFEEF6FF);
const Color kAccent200 = Color(0xFFD6EBFF);
const Color kAccent300 = Color(0xFFB5D9FD);
const Color kAccent400 = Color(0xFF94BCE3);
const Color kAccent500 = Color(0xFF749DC4);
const Color kAccent600 = Color(0xFF597EA3);
const Color kAccent700 = Color(0xFF416180);
const Color kAccent800 = Color(0xFF2C455D);
const Color kAccent900 = Color(0xFF1D2D3D);
const Color kAccent2100 = Color(0xFFEEF6FF);
const Color kAccent2300 = Color(0xFFBDD8F2);
const Color kAccent2400 = Color(0xFF9EBBD8);
const Color kAccent2500 = Color(0xFF7E9CB8);
const Color kAccent2700 = Color(0xFF486077);
const Color kAccent2900 = Color(0xFF1F2D3A);

/// 잉크(밝기 무관 고정색) — 카메라/썸네일 바탕, 스크림·그림자 기저색.
const Color kInkDeep = Color(0xFF131518);
const Color kInkDeep2 = Color(0xFF1D2023);
const Color kInkShadow = Color(0xFF1D1F20);
const Color kInkCardShadow = Color(0xFF2B2B2D);

/// 모션
const Duration kPageTransitionDuration = Duration(milliseconds: 380);
const Curve kPageTransitionCurve = Curves.easeOutCubic;
const Duration kCardEnterDuration = Duration(milliseconds: 450);
const Duration kFadeShortDuration = Duration(milliseconds: 220);

/// 코너 반경
const double kRadiusSmall = 12;
const double kRadiusMedium = 16;
const double kRadiusLarge = 22;
const double kRadiusXL = 28;

/// Spacing
const double kSpaceXS = 4;
const double kSpaceS = 8;
const double kSpaceM = 12;
const double kSpaceL = 16;
const double kSpaceXL = 24;
const double kSpaceXXL = 32;


ColorScheme _buildScheme(Brightness brightness) {
  // M3 Dynamic Scheme 의 ColorScheme.fromSeed 가 secondary/tertiary 까지 자동 생성하지만
  // 직접 적당한 색을 secondary/tertiary 에 끼워넣어 다양성 확보.
  final base = ColorScheme.fromSeed(
    seedColor: brandSeed,
    brightness: brightness,
  );
  final secondary = ColorScheme.fromSeed(seedColor: secondarySeed, brightness: brightness);
  final tertiary = ColorScheme.fromSeed(seedColor: tertiarySeed, brightness: brightness);

  // 다크 모드의 primary 는 M3 기본이 시드를 tone 80 으로 밝힌 파스텔톤이 됨.
  // 브랜드 일관성을 위해 라이트와 동일한 brandSeed(Steel Blue) 로 통일.
  final isDark = brightness == Brightness.dark;
  final primary = isDark ? brandSeed : base.primary;
  final onPrimary = isDark ? Colors.white : base.onPrimary;

  return base.copyWith(
    primary: primary,
    onPrimary: onPrimary,
    secondary: secondary.primary,
    onSecondary: secondary.onPrimary,
    secondaryContainer: secondary.primaryContainer,
    // 다크의 on*Container 는 연두 톤이라 텍스트 대비가 낮음 → 흰색으로 통일
    onSecondaryContainer:
        isDark ? Colors.white : secondary.onPrimaryContainer,
    onPrimaryContainer: isDark ? Colors.white : base.onPrimaryContainer,
    tertiary: tertiary.primary,
    onTertiary: tertiary.onPrimary,
    tertiaryContainer: tertiary.primaryContainer,
    onTertiaryContainer: tertiary.onPrimaryContainer,
  );
}


TextTheme _textTheme(Brightness brightness) {
  // 라이트는 거의 순수 검정, 다크는 거의 순수 흰색으로 강한 대비.
  final base = brightness == Brightness.light
      ? Typography.material2021().black
      : Typography.material2021().white;

  final emphasis = brightness == Brightness.light
      ? const Color(0xFF0F1419)  // 거의 검정 (Twitter-style 어두운 잉크)
      : const Color(0xFFFFFFFF); // 순수 흰색

  TextStyle? bold(TextStyle? t) => t?.copyWith(
        fontWeight: FontWeight.w700,
        letterSpacing: -0.5,
        color: emphasis,
      );
  TextStyle? semi(TextStyle? t) => t?.copyWith(
        fontWeight: FontWeight.w600,
        color: emphasis,
      );

  return base.copyWith(
    displayLarge: bold(base.displayLarge),
    displayMedium: bold(base.displayMedium),
    displaySmall: bold(base.displaySmall),
    headlineLarge: bold(base.headlineLarge),
    headlineMedium: bold(base.headlineMedium),
    headlineSmall: semi(base.headlineSmall),
    titleLarge: semi(base.titleLarge),
    titleMedium: semi(base.titleMedium),
    titleSmall: semi(base.titleSmall),
    labelLarge: semi(base.labelLarge),
    bodyLarge: base.bodyLarge?.copyWith(height: 1.5, color: emphasis),
    bodyMedium: base.bodyMedium?.copyWith(height: 1.5, color: emphasis),
    bodySmall: base.bodySmall?.copyWith(height: 1.45),
  );
}


ThemeData _buildTheme(Brightness brightness) {
  final scheme = _buildScheme(brightness);
  final textTheme = _textTheme(brightness);
  final isLight = brightness == Brightness.light;

  // 라이트: 시안 배경 그레이 scaffold + 흰색 card 단계
  // 다크: 깊은 잉크 scaffold + 살짝 밝은 card
  final scaffoldColor = isLight
      ? const Color(0xFFF2F2F3)
      : const Color(0xFF0F1419);

  return ThemeData(
    colorScheme: scheme,
    useMaterial3: true,
    fontFamily: kBodyFontFamily,
    textTheme: textTheme,
    scaffoldBackgroundColor: scaffoldColor,
    canvasColor: scaffoldColor,
    appBarTheme: AppBarTheme(
      backgroundColor: scaffoldColor,
      foregroundColor: scheme.onSurface,
      surfaceTintColor: Colors.transparent,
      centerTitle: false,
      elevation: 0,
      scrolledUnderElevation: 0.6,
      titleTextStyle: textTheme.titleLarge,
      iconTheme: IconThemeData(color: scheme.onSurface),
      systemOverlayStyle: SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: isLight ? Brightness.dark : Brightness.light,
        statusBarBrightness: isLight ? Brightness.light : Brightness.dark,
        systemNavigationBarColor: scaffoldColor,
        systemNavigationBarIconBrightness:
            isLight ? Brightness.dark : Brightness.light,
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(54),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(kRadiusMedium),
        ),
        textStyle: textTheme.labelLarge?.copyWith(fontSize: 15),
      ),
    ),
    // 다크에서 primary(진초록) 텍스트는 대비가 낮음 → 흰색으로 대체
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: scheme.brightness == Brightness.dark
            ? Colors.white
            : scheme.primary,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(54),
        foregroundColor: scheme.brightness == Brightness.dark
            ? Colors.white
            : scheme.primary,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(kRadiusMedium),
        ),
        textStyle: textTheme.labelLarge?.copyWith(fontSize: 15),
        side: BorderSide(color: scheme.outlineVariant, width: 1.4),
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(
        foregroundColor: scheme.onSurface,
      ),
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: isLight ? Colors.white : const Color(0xFF1A2027),
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        side: BorderSide(
          color: scheme.outlineVariant.withValues(alpha: isLight ? 0.5 : 0.4),
        ),
        borderRadius: BorderRadius.circular(kRadiusLarge),
      ),
      margin: EdgeInsets.zero,
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: isLight ? const Color(0xFFF1F3F5) : const Color(0xFF1A2027),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadiusMedium),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadiusMedium),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadiusMedium),
        borderSide: BorderSide(color: scheme.primary, width: 2),
      ),
      contentPadding: const EdgeInsets.symmetric(
        horizontal: kSpaceL, vertical: kSpaceL,
      ),
    ),
    floatingActionButtonTheme: FloatingActionButtonThemeData(
      backgroundColor: scheme.primary,
      foregroundColor: scheme.onPrimary,
      elevation: 3,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadiusMedium),
      ),
    ),
    chipTheme: ChipThemeData(
      backgroundColor: isLight
          ? const Color(0xFFF1F3F5)
          : const Color(0xFF1A2027),
      labelStyle: textTheme.labelMedium,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadiusSmall),
        side: BorderSide(
          color: scheme.outlineVariant.withValues(alpha: isLight ? 0.4 : 0.3),
        ),
      ),
      side: BorderSide.none,
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadiusMedium),
      ),
      backgroundColor: scheme.inverseSurface,
      contentTextStyle: TextStyle(color: scheme.onInverseSurface),
    ),
    bottomSheetTheme: BottomSheetThemeData(
      backgroundColor: scaffoldColor,
      surfaceTintColor: Colors.transparent,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(kRadiusXL)),
      ),
      showDragHandle: true,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: isLight ? Colors.white : const Color(0xFF1A2027),
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadiusXL),
      ),
    ),
    listTileTheme: ListTileThemeData(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadiusMedium),
      ),
      contentPadding: const EdgeInsets.symmetric(
        horizontal: kSpaceL, vertical: kSpaceXS,
      ),
      iconColor: scheme.onSurface,
    ),
    dividerTheme: DividerThemeData(
      color: scheme.outlineVariant.withValues(alpha: 0.4),
      thickness: 1,
      space: 1,
    ),
    pageTransitionsTheme: const PageTransitionsTheme(
      builders: {
        TargetPlatform.android: PredictiveBackPageTransitionsBuilder(),
      },
    ),
  );
}


ThemeData buildLightTheme() => _buildTheme(Brightness.light);
ThemeData buildDarkTheme() => _buildTheme(Brightness.dark);
