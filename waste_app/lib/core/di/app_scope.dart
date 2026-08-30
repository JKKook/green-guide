/// 앱 전역 의존 접근점 — 설정·기록 DB·분류 서비스·API 클라이언트.
///
/// 화면/위젯은 `SettingsStore()` 같은 직접 생성 대신 여기서 가져온다.
/// 라이브러리(get_it 등) 없이 정적 필드로 충분한 규모라 단순하게 둔다.
/// 테스트는 각 저장소의 가짜(SharedPreferences 목, sqflite ffi)를 주입하므로
/// 인스턴스 교체 기능은 두지 않는다.
library;

import '../../api/api_client.dart';
import '../../data/history_repository.dart';
import '../../data/settings_store.dart';
import '../../services/prediction_service.dart';

class AppScope {
  AppScope._();

  static final SettingsStore settings = SettingsStore();
  static final HistoryRepository history = HistoryRepository();
  static final PredictionService prediction = PredictionService(settings);

  /// 현재 설정된 서버를 가리키는 API 클라이언트.
  /// baseUrl 은 개발자 옵션에서 바뀔 수 있어 호출 시점마다 설정에서 읽는다.
  static Future<WasteApiClient> api({
    Duration timeout = const Duration(seconds: 30),
  }) async =>
      WasteApiClient(baseUrl: await settings.getApiUrl(), timeout: timeout);
}
