import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../core/di/app_scope.dart';
import '../theme/app_theme.dart';
import 'main_shell.dart';
import 'onboarding_screen.dart';

/// 앱 진입점 — 첫 실행 여부에 따라 Onboarding 또는 Home 으로 라우팅.
/// 네이티브 스플래시가 사라진 직후 표시되는 Flutter 스플래시.
/// 자체 fade + scale + 펄스 애니메이션 후 다음 화면으로 전환.
class SplashRouter extends StatefulWidget {
  const SplashRouter({super.key});

  @override
  State<SplashRouter> createState() => _SplashRouterState();
}

class _SplashRouterState extends State<SplashRouter>
    with TickerProviderStateMixin {
  late final AnimationController _entryController;
  late final AnimationController _pulseController;
  late final Animation<double> _fadeAnim;
  late final Animation<double> _scaleAnim;
  late final Animation<double> _pulseAnim;
  late final Animation<double> _titleFadeAnim;

  bool _navigating = false;

  @override
  void initState() {
    super.initState();

    // 1) 진입 애니메이션 — fade in + 살짝 튕기는 scale
    _entryController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _fadeAnim = CurvedAnimation(parent: _entryController, curve: Curves.easeOut);
    _scaleAnim = Tween<double>(begin: 0.55, end: 1.0).animate(
      CurvedAnimation(parent: _entryController, curve: Curves.easeOutBack),
    );
    _titleFadeAnim = CurvedAnimation(
      parent: _entryController,
      curve: const Interval(0.55, 1.0, curve: Curves.easeOut),
    );

    // 2) 펄스 애니메이션 — 진입 후 부드럽게 호흡
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );
    _pulseAnim = Tween<double>(begin: 1.0, end: 1.06).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _entryController.forward().then((_) {
      _pulseController.repeat(reverse: true);
    });

    // 3) 최소 1.6초 표시 후 라우팅 결정 (애니메이션 + 살짝의 여운)
    Future.delayed(const Duration(milliseconds: 1600), _decideRoute);
  }

  @override
  void dispose() {
    _entryController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  Future<void> _decideRoute() async {
    if (_navigating || !mounted) return;
    _navigating = true;

    final done = await AppScope.settings.isOnboardingDone();
    if (!mounted) return;

    Navigator.of(context).pushReplacement(PageRouteBuilder(
      transitionDuration: const Duration(milliseconds: 380),
      pageBuilder: (_, _, _) =>
          done ? const MainShell() : const OnboardingScreen(),
      transitionsBuilder: (_, anim, _, child) => FadeTransition(
        opacity: CurvedAnimation(parent: anim, curve: Curves.easeOut),
        child: child,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light,
        systemNavigationBarColor: brandSeed, // 메인 컬러 — 스플래시 배경과 동일
        systemNavigationBarIconBrightness: Brightness.light,
      ),
      child: Scaffold(
        backgroundColor: brandSeed,
        body: Center(
          child: AnimatedBuilder(
            animation: Listenable.merge([_entryController, _pulseController]),
            builder: (context, _) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // 로고: fade + scale + pulse 합성
                  Opacity(
                    opacity: _fadeAnim.value,
                    child: Transform.scale(
                      scale: _scaleAnim.value *
                          (_entryController.isCompleted ? _pulseAnim.value : 1.0),
                      child: const Image(
                        image: AssetImage('assets/splash/logo.png'),
                        width: 160,
                        height: 160,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  // 앱 이름 — 늦게 fade in
                  Opacity(
                    opacity: _titleFadeAnim.value,
                    child: const Text(
                      '그린가이드',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -0.4,
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Opacity(
                    opacity: _titleFadeAnim.value * 0.7,
                    child: const Text(
                      '분리수거 AI 가이드',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }
}
