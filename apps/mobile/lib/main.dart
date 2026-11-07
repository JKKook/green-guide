import 'dart:async';

import 'package:flutter/material.dart';

import 'app.dart';
import 'data/haptics.dart';
import 'data/licenses.dart';
import 'data/settings_store.dart';
import 'services/class_loader.dart';
import 'services/server_warmup.dart';
import 'services/telemetry.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // 크래시 수집을 가장 먼저 — 이후 초기화 중 오류도 잡히게. 미구성이면 조용히 건너뜀.
  await Telemetry.init();
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
