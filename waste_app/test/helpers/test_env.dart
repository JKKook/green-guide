/// 위젯 테스트 공용 환경 — 플러그인(SharedPreferences·sqflite·path_provider·
/// package_info) 을 호스트에서 동작하는 가짜로 대체한다.
library;

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:waste_app/theme/app_theme.dart';

class _FakePathProvider extends PathProviderPlatform
    with MockPlatformInterfaceMixin {
  _FakePathProvider(this.root);
  final String root;

  @override
  Future<String?> getApplicationDocumentsPath() async => root;
  @override
  Future<String?> getTemporaryPath() async => root;
  @override
  Future<String?> getApplicationSupportPath() async => root;
}

/// 테스트마다 격리된 임시 디렉토리를 만들고 플러그인을 가짜로 바꾼다.
/// 반환된 디렉토리는 tearDown 에서 지운다.
Future<Directory> setUpTestEnv({Map<String, Object> prefs = const {}}) async {
  TestWidgetsFlutterBinding.ensureInitialized();
  SharedPreferences.setMockInitialValues(prefs);
  PackageInfo.setMockInitialValues(
    appName: 'waste_app',
    packageName: 'test',
    version: '1.0.0-test',
    buildNumber: '1',
    buildSignature: '',
  );
  sqfliteFfiInit();
  databaseFactory = databaseFactoryFfi;
  final dir = await Directory.systemTemp.createTemp('waste_app_test_');
  PathProviderPlatform.instance = _FakePathProvider(dir.path);
  return dir;
}

/// 앱과 동일한 테마·로케일로 감싼 MaterialApp.
Widget wrapApp(Widget home) => MaterialApp(
  theme: buildLightTheme(),
  locale: const Locale('ko'),
  home: home,
);

/// DB(isolate)·파일 I/O 같은 실제 비동기가 끝나도록 잠시 실시간을 흘린 뒤 한 프레임 그린다.
/// (testWidgets 의 가짜 시계는 isolate 응답을 진행시키지 못해 그냥 pump 하면 영원히 대기)
Future<void> settleIo(
  WidgetTester tester, [
  Duration wait = const Duration(milliseconds: 400),
]) async {
  await tester.runAsync(() => Future<void>.delayed(wait));
  await tester.pump();
}
