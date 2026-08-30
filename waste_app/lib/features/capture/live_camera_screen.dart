/// 스마트 촬영 — 시안 16b: 5초 카운트다운 자동 캡처 + 전용 프로그레스 바.
/// 카운트다운이 끝나면 흔들림이 잦아든 순간(안정도 감지)에 촬영한다.
library;

import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../data/haptics.dart';
import '../../data/image_prep.dart';
import '../../services/stability_detector.dart';
import '../../theme/app_theme.dart';
import '../result/result_modal.dart';
import 'capture_entry_sheet.dart' show pickFromGalleryAndAnalyze;
import 'widgets/camera_overlays.dart';

const Duration _kCountdown = Duration(seconds: 5);
const Duration _kStableGrace = Duration(seconds: 3);

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}


class _LiveCameraScreenState extends State<LiveCameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = const [];
  CameraLensDirection _lens = CameraLensDirection.back;
  bool _torch = false;

  late final StabilityDetector _stability;
  double _stabilityProgress = 0.0;

  Timer? _tick;
  Timer? _graceTimer;
  Duration _elapsed = Duration.zero;
  bool _armed = false; // 카운트다운 종료 — 안정되는 즉시 촬영
  bool _capturing = false;
  bool _paused = false;

  String? _initError;
  bool _isPermissionError = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _stability = StabilityDetector(
      onStable: _onStable,
      window: const Duration(seconds: 1),
    );
    _stability.progress.listen((p) {
      if (mounted) setState(() => _stabilityProgress = p);
    });
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      if (_cameras.isEmpty) _cameras = await availableCameras();
      if (!mounted) return;
      if (_cameras.isEmpty) {
        setState(() => _initError = '사용할 수 있는 카메라가 없어요');
        return;
      }
      final cam = _cameras.firstWhere(
        (c) => c.lensDirection == _lens,
        orElse: () => _cameras.first,
      );
      final controller = CameraController(
        cam,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _controller = controller);
      _startCountdown();
    } on CameraException catch (e) {
      if (!mounted) return;
      final code = e.code.toLowerCase();
      final isPermission = code.contains('permission') ||
          code.contains('denied') ||
          code.contains('access');
      setState(() {
        _isPermissionError = isPermission;
        _initError = isPermission
            ? '카메라 권한이 필요해요'
            : '카메라를 열지 못했어요 · 다시 시도해 주세요';
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _initError = '카메라를 열지 못했어요 · 다시 시도해 주세요');
    }
  }

  // ── 카운트다운 ───────────────────────────────────────────────

  void _startCountdown() {
    _tick?.cancel();
    _graceTimer?.cancel();
    _armed = false;
    _elapsed = Duration.zero;
    _stability.reset();
    _stability.start();
    _tick = Timer.periodic(const Duration(milliseconds: 50), (timer) {
      if (!mounted || _paused) return;
      setState(() => _elapsed += const Duration(milliseconds: 50));
      if (_elapsed >= _kCountdown) {
        timer.cancel();
        _armCapture();
      }
    });
  }

  /// 5초 경과 — 이미 안정적이면 즉시, 아니면 안정 순간(최대 3초 대기)에 촬영.
  void _armCapture() {
    if (_capturing || _paused) return;
    if (_stabilityProgress >= 0.6) {
      _capture(autoTriggered: true);
      return;
    }
    setState(() => _armed = true);
    Haptics.light();
    _graceTimer = Timer(_kStableGrace, () {
      if (mounted && _armed && !_capturing) _capture(autoTriggered: true);
    });
  }

  void _onStable() {
    if (_armed && !_capturing && !_paused) {
      _graceTimer?.cancel();
      _capture(autoTriggered: true);
    }
  }

  int get _secondsLeft {
    final left = _kCountdown - _elapsed;
    return left.isNegative ? 0 : (left.inMilliseconds / 1000).ceil();
  }

  // ── 캡처 ───────────────────────────────────────────────────

  Future<void> _capture({bool autoTriggered = false}) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (controller.value.isTakingPicture || _capturing) return;

    setState(() {
      _capturing = true;
      _armed = false;
    });
    Haptics.heavy();
    _tick?.cancel();
    _graceTimer?.cancel();
    _stability.stop();

    try {
      final shot = await controller.takePicture();
      if (!mounted) return;
      _paused = true;
      if (_torch) {
        await controller.setFlashMode(FlashMode.off);
        _torch = false;
      }
      if (!mounted) return;

      // 크롭 판단은 서버가 u2netp saliency 로 수행하므로 프레임 전체를 보내되,
      // 업로드량을 갤러리 경로와 같은 크기(긴 변 1600px)로 줄인다.
      final upload = await prepareForUpload(File(shot.path));
      if (!mounted) return;
      final closeCamera = await showResultModal(
        context,
        upload,
        isSmartCapture: true,
      );

      if (!mounted) return;
      _paused = false;
      if (closeCamera != false) {
        Navigator.of(context).pop();
        return;
      }
      // 분석 취소 → 다시 촬영: 카운트다운 재시작
      _startCountdown();
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<void> _toggleTorch() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    Haptics.selection();
    try {
      await controller.setFlashMode(_torch ? FlashMode.off : FlashMode.torch);
      if (mounted) setState(() => _torch = !_torch);
    } catch (_) {}
  }

  Future<void> _flipCamera() async {
    if (_cameras.length < 2 || _capturing) return;
    Haptics.selection();
    _tick?.cancel();
    _graceTimer?.cancel();
    _stability.stop();
    final old = _controller;
    setState(() {
      _controller = null;
      _torch = false;
      _lens = _lens == CameraLensDirection.back
          ? CameraLensDirection.front
          : CameraLensDirection.back;
    });
    await old?.dispose();
    await _initCamera();
  }

  Future<void> _openGallery() async {
    if (_capturing) return;
    _paused = true;
    _tick?.cancel();
    _graceTimer?.cancel();
    _stability.stop();
    await pickFromGalleryAndAnalyze(context);
    if (!mounted) return;
    _paused = false;
    _startCountdown();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _tick?.cancel();
      _graceTimer?.cancel();
      _stability.stop();
      // dispose 한 컨트롤러를 build 가 계속 참조하면 "disposed CameraController"
      // 예외가 난다 — 참조를 먼저 끊고 미리보기를 준비 상태로 되돌린다.
      if (mounted) setState(() => _controller = null);
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      // 결과 모달이 떠 있는 동안(_paused)에는 카운트다운을 재시작하지 않는다.
      if (!_paused) _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _tick?.cancel();
    _graceTimer?.cancel();
    _stability.dispose();
    _controller?.dispose();
    super.dispose();
  }

  // ── UI ──────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final ready = controller != null && controller.value.isInitialized;
    final progress =
        (_elapsed.inMilliseconds / _kCountdown.inMilliseconds).clamp(0.0, 1.0);

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light,
        systemNavigationBarColor: kInkDeep,
        systemNavigationBarIconBrightness: Brightness.light,
      ),
      child: Scaffold(
        backgroundColor: kInkDeep,
        body: SafeArea(
          child: Column(
            children: [
              // 헤더 — 닫기 · 제목 · 플래시
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
                child: Row(
                  children: [
                    GlassButton(
                      size: 38,
                      semanticLabel: '촬영 닫기',
                      onTap: () => Navigator.of(context).pop(),
                      child: const Icon(Icons.close,
                          size: 17, color: kNeutral100),
                    ),
                    const Expanded(
                      child: Text(
                        '스마트 촬영',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: kNeutral100,
                        ),
                      ),
                    ),
                    GlassButton(
                      size: 38,
                      semanticLabel: _torch ? '플래시 끄기' : '플래시 켜기',
                      onTap: _toggleTorch,
                      active: _torch,
                      child: Icon(
                        _torch ? Icons.bolt : Icons.bolt_outlined,
                        size: 17,
                        color: _torch ? kAccent300 : kNeutral100,
                      ),
                    ),
                  ],
                ),
              ),
              // 뷰파인더 카드
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 2, 16, 0),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(24),
                    child: Container(
                      color: kInkDeep2,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          if (ready)
                            FittedBox(
                              fit: BoxFit.cover,
                              clipBehavior: Clip.hardEdge,
                              child: SizedBox(
                                width:
                                    controller.value.previewSize?.height ?? 1,
                                height:
                                    controller.value.previewSize?.width ?? 1,
                                child: CameraPreview(controller),
                              ),
                            )
                          else if (_initError == null)
                            const Center(
                              child: CircularProgressIndicator(
                                  color: kNeutral100),
                            ),
                          // 코너 브래킷
                          for (final (a, top, left) in const [
                            (Alignment.topLeft, true, true),
                            (Alignment.topRight, true, false),
                            (Alignment.bottomLeft, false, true),
                            (Alignment.bottomRight, false, false),
                          ])
                            Align(
                              alignment: a,
                              child: Padding(
                                padding: const EdgeInsets.all(14),
                                child: CaptureCorner(top: top, left: left),
                              ),
                            ),
                          // 카운트다운 + 힌트
                          if (ready && !_capturing)
                            IgnorePointer(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  if (!_armed)
                                    Text(
                                      '${_secondsLeft == 0 ? 1 : _secondsLeft}',
                                      style: const TextStyle(
                                        fontSize: 104,
                                        fontWeight: FontWeight.w600,
                                        height: 1,
                                        color: kNeutral100,
                                        shadows: [
                                          Shadow(
                                            color: Color(0x8C000000),
                                            blurRadius: 24,
                                            offset: Offset(0, 2),
                                          ),
                                        ],
                                      ),
                                    ),
                                  const SizedBox(height: 8),
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 14, vertical: 7),
                                    decoration: BoxDecoration(
                                      color: kInkDeep.withValues(alpha: 0.65),
                                      borderRadius: BorderRadius.circular(999),
                                    ),
                                    child: Text(
                                      _armed
                                          ? '흔들림을 줄여주세요 — 곧 촬영해요'
                                          : '품목을 프레임 중앙에 담아주세요',
                                      style: const TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w600,
                                        color: kNeutral100,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          // 캡처 플래시
                          if (_capturing)
                            TweenAnimationBuilder<double>(
                              tween: Tween(begin: 1.0, end: 0.0),
                              duration: const Duration(milliseconds: 300),
                              builder: (_, v, _) => Container(
                                color: Colors.white.withValues(alpha: v * 0.7),
                              ),
                            ),
                          // 에러
                          if (_initError != null)
                            CameraErrorOverlay(
                              message: _initError!,
                              isPermission: _isPermissionError,
                              onRetry: () {
                                setState(() {
                                  _initError = null;
                                  _isPermissionError = false;
                                });
                                _initCamera();
                              },
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              // 스마트 캡처 프로그레스
              Padding(
                padding: const EdgeInsets.fromLTRB(24, 18, 24, 6),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.timer_outlined,
                            size: 14, color: kAccent400),
                        const SizedBox(width: 7),
                        const Text(
                          '스마트 캡처 · 5초 뒤 자동 촬영',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: Color(0xE6FFFFFF),
                          ),
                        ),
                        const Spacer(),
                        Text(
                          _armed
                              ? '안정되면 촬영'
                              : ready
                                  ? '$_secondsLeft초 남음'
                                  : '준비 중',
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: kAccent400,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 9),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: LinearProgressIndicator(
                        value: _armed ? 1 : progress,
                        minHeight: 6,
                        backgroundColor: Colors.white.withValues(alpha: 0.14),
                        color: kAccent400,
                      ),
                    ),
                  ],
                ),
              ),
              // 컨트롤 — 갤러리 · 셔터 · 전환
              Padding(
                padding: EdgeInsets.fromLTRB(
                  44,
                  14,
                  44,
                  18 + MediaQuery.viewPaddingOf(context).bottom * 0,
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    GlassButton(
                      size: 46,
                      radius: 14,
                      outlined: true,
                      semanticLabel: '갤러리에서 사진 선택',
                      onTap: _openGallery,
                      child: const Icon(Icons.image_outlined,
                          size: 20, color: kNeutral100),
                    ),
                    Semantics(
                      button: true,
                      label: '지금 촬영',
                      child: GestureDetector(
                      onTap: (ready && !_capturing) ? _capture : null,
                      child: Container(
                        width: 72,
                        height: 72,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(color: kNeutral100, width: 4),
                        ),
                        padding: const EdgeInsets.all(kSpaceXS),
                        child: Container(
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: (ready && !_capturing)
                                ? kNeutral100
                                : kNeutral500,
                          ),
                        ),
                      ),
                      ),
                    ),
                    GlassButton(
                      size: 46,
                      outlined: true,
                      semanticLabel: '전면·후면 카메라 전환',
                      onTap: _flipCamera,
                      child: const Icon(Icons.cameraswitch_outlined,
                          size: 19, color: kNeutral100),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
