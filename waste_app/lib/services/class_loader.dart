/// 앱 시작 시 /labels 호출해 ClassRegistry 캐시 채움.
///
/// 성공 시 디스크(SharedPreferences)에도 저장 → 다음 오프라인 부팅 때 재사용.
/// 디스크 캐시까지 없을 때만 하드코딩된 _fallbackClasses 로 폴백한다.
/// 덕분에 한 번이라도 온라인이었으면 클래스가 N개로 늘어나도 코드 수정 없이
/// 오프라인에서 최신 전체 클래스 메타를 그대로 사용한다.
library;

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../core/di/app_scope.dart';
import '../data/settings_store.dart';
import '../data/waste_info.dart';


class ClassLoader {
  final SettingsStore _settings = AppScope.settings;
  final Duration timeout;

  ClassLoader({this.timeout = const Duration(seconds: 8)});

  /// 서버에서 클래스 목록 + 메타 로드 → 성공 시 디스크 캐시.
  /// 실패(오프라인/서버다운) 시 디스크 캐시 → 그것도 없으면 false (하드코딩 fallback).
  Future<bool> loadFromServer() async {
    try {
      final baseUrl = await _settings.getApiUrl();
      final uri = Uri.parse(
        '${baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl}/labels',
      );
      final res = await http.get(uri).timeout(timeout);
      if (res.statusCode != 200) return _loadFromCache();

      final body = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      final classes = body['classes'] as List?;
      if (classes == null) return _loadFromCache();

      WasteClassRegistry.setFromApi(classes.cast<Map<String, dynamic>>());
      await _saveCache(classes);
      return true;
    } catch (_) {
      return _loadFromCache();
    }
  }

  /// 마지막으로 성공한 /labels 응답을 디스크에서 복원.
  Future<bool> _loadFromCache() async {
    try {
      final cached = await _settings.getCachedClassesJson();
      if (cached == null) return false;
      final classes = (jsonDecode(cached) as List).cast<Map<String, dynamic>>();
      WasteClassRegistry.setFromApi(classes);
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<void> _saveCache(List<dynamic> classes) async {
    try {
      await _settings.setCachedClassesJson(jsonEncode(classes));
    } catch (_) {
      // 캐시 저장 실패는 무시 — 다음 부팅에 다시 시도
    }
  }
}
