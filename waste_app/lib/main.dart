import 'dart:async';

import 'package:flutter/material.dart';

import 'app.dart';
import 'data/haptics.dart';
import 'data/licenses.dart';
import 'data/settings_store.dart';
import 'services/class_loader.dart';
import 'services/server_warmup.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
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
