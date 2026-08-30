/// 디바이스 안정도 측정 서비스.
///
/// 가속도계 데이터의 magnitude 가 중력(9.8m/s²)에서 [threshold] 이내로
/// [window] 시간만큼 유지되면 "안정 상태"로 판단.
library;

import 'dart:async';
import 'dart:math';

import 'package:sensors_plus/sensors_plus.dart';


class StabilityDetector {
  /// 안정으로 간주하는 최대 가속도 편차 (m/s²)
  /// 0.4 = 손에 들고 거의 가만히 있을 때 수준 (가벼운 떨림 허용)
  final double threshold;

  /// 강한 흔들림 임계 (m/s²). 이 이상은 grace window 무시하고 **즉시** 리셋.
  /// 물체가 미세하게 움직이는 건 OK 하지만 카메라가 심하게 흔들리면 촬영 의미 없음 →
  /// 사용자에게 "다시 안정시켜" 라는 시각 신호 (progress 0으로 떨어짐) 즉시 전달.
  final double severeThreshold;

  /// 안정이 유지되어야 하는 시간
  final Duration window;

  /// 순간 흔들림 허용 시간 — 이 시간보다 짧은 흔들림은 진행을 리셋하지 않음.
  /// 미세한 떨림(샘플 수준의 순간 스파이크)만 흡수하기 위한 짧은 grace.
  /// 단, severeThreshold 를 넘으면 graceWindow 무시.
  final Duration graceWindow;

  /// progress(0.0~1.0) 를 emit 하는 stream. 1.0 도달 시 onStable 호출.
  final void Function() onStable;

  StreamSubscription<AccelerometerEvent>? _sub;
  final StreamController<double> _progressController = StreamController.broadcast();
  DateTime? _stableStartedAt;
  DateTime? _unstableSince;   // 연속 흔들림 시작 시각 (grace 판정용)
  double _lastProgress = 0.0;
  bool _triggered = false;

  Stream<double> get progress => _progressController.stream;

  StabilityDetector({
    required this.onStable,
    this.threshold = 0.4,
    this.severeThreshold = 1.5,
    this.window = const Duration(seconds: 3),
    this.graceWindow = const Duration(milliseconds: 250),
  });

  void start() {
    // 이전 구독을 항상 정리(중복 listen 으로 _onAccel 이 2배 호출되는 문제 방지).
    _sub?.cancel();
    _sub = null;
    _triggered = false;
    _stableStartedAt = null;
    _unstableSince = null;
    _lastProgress = 0.0;
    _sub = accelerometerEventStream(
      samplingPeriod: const Duration(milliseconds: 50),
    ).listen(_onAccel);
  }

  void _onAccel(AccelerometerEvent e) {
    if (_triggered) return;
    final magnitude = sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
    final delta = (magnitude - 9.81).abs();
    final now = DateTime.now();

    if (delta < threshold) {
      // 안정 — 흔들림 추적 해제하고 진행률 누적.
      _unstableSince = null;
      _stableStartedAt ??= now;
      final elapsed = now.difference(_stableStartedAt!).inMilliseconds;
      final prog = (elapsed / window.inMilliseconds).clamp(0.0, 1.0);
      _emitProgress(prog);

      if (prog >= 1.0) {
        _triggered = true;
        onStable();
      }
    } else if (delta >= severeThreshold) {
      // 강한 흔들림 — grace window 무시하고 즉시 리셋.
      // 카메라 자체가 흔들리는 케이스 (조준 흔들림·손 격렬한 떨림 등) → 촬영 의미 없음.
      if (_stableStartedAt != null) {
        _stableStartedAt = null;
        _unstableSince = null;
        _emitProgress(0.0);
      }
    } else {
      // 약한 흔들림 (threshold ≤ delta < severeThreshold)
      // grace window 보다 짧으면 진행을 유지(리셋하지 않음).
      if (_stableStartedAt == null) return;  // 아직 시작 전
      _unstableSince ??= now;
      final unstableMs = now.difference(_unstableSince!).inMilliseconds;
      if (unstableMs >= graceWindow.inMilliseconds) {
        // 지속적 흔들림 → 진행 리셋
        _stableStartedAt = null;
        _unstableSince = null;
        _emitProgress(0.0);
      }
      // grace window 내의 순간 떨림은 무시 (_stableStartedAt·진행률 유지)
    }
  }

  void _emitProgress(double prog) {
    if (prog == _lastProgress) return;
    _lastProgress = prog;
    if (!_progressController.isClosed) _progressController.add(prog);
  }

  /// 외부에서 강제로 트리거된 후 재시작
  void reset() {
    _triggered = false;
    _stableStartedAt = null;
    _unstableSince = null;
    _emitProgress(0.0);
  }

  void stop() {
    _sub?.cancel();
    _sub = null;
  }

  Future<void> dispose() async {
    stop();
    await _progressController.close();
  }
}
