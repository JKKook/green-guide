/// HF Spaces 무료 티어는 무요청 상태가 이어지면 절전으로 내려가고, 깨어나는 데
/// 수 분이 걸린다. 앱 부팅 직후 `/health` 를 한 번 깨워두면 사용자가 촬영을
/// 마칠 즈음엔 서버가 준비된 상태가 된다. 실패해도 무해(분석 때 다시 시도).
library;

import '../core/di/app_scope.dart';


class ServerWarmup {
  const ServerWarmup._();

  static bool _started = false;

  /// 앱 세션당 한 번만 — 응답을 기다리지 않고 백그라운드로 던져둔다.
  static Future<void> ping() async {
    if (_started) return;
    _started = true;
    try {
      final client = await AppScope.api(timeout: const Duration(minutes: 3));
      await client.isHealthy();
    } catch (_) {
      // 절전 해제 실패 — 실제 분석 요청에서 다시 시도된다.
    }
  }
}
