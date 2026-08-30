import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:waste_app/core/di/app_scope.dart';
import 'package:waste_app/data/settings_store.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('api() 는 호출 시점의 설정 URL 을 쓴다 (개발자 옵션에서 바뀔 수 있음)', () async {
    SharedPreferences.setMockInitialValues({});
    expect((await AppScope.api()).baseUrl, SettingsStore.defaultApiUrl);

    await AppScope.settings.setApiUrl('http://localhost:8000');
    expect((await AppScope.api()).baseUrl, 'http://localhost:8000');
  });

  test('settings/history/prediction 은 단일 인스턴스', () {
    expect(identical(AppScope.settings, AppScope.settings), isTrue);
    expect(identical(AppScope.history, AppScope.history), isTrue);
    expect(identical(AppScope.prediction, AppScope.prediction), isTrue);
  });
}
