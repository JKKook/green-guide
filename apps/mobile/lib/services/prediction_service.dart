/// 분류 요청 라우팅 — 베타는 클라우드(waste-api) 전용.
///
/// 호출 측 코드(ResultModal 등) 는 이 클래스의 predict() 만 쓰면 됨.
/// (온디바이스 ONNX 경로는 Play 16KB 페이지 요건 미충족(onnxruntime 1.4.1)으로
///  2026-08-29 베타에서 제외 — 정확도도 서버 경로가 앙상블·TTA·객체 분리로 우위.)
library;

import 'dart:io';

import '../api/api_client.dart';
import '../api/models.dart';
import '../core/log.dart';
import '../data/settings_store.dart';


class PredictionService {
  final SettingsStore _settings;

  PredictionService(this._settings);

  /// [centered] 는 구버전 서버 폴백 시 `/predict-centered` 선택에만 쓰인다.
  Future<Prediction> predict(File image,
      {bool centered = false, UploadProgress? onUploadProgress}) {
    return _cloudPredict(image,
        centered: centered, onUploadProgress: onUploadProgress);
  }

  Future<Prediction> _cloudPredict(File image,
      {Duration? timeout,
      bool centered = false,
      UploadProgress? onUploadProgress}) async {
    final baseUrl = await _settings.getApiUrl();
    final client = WasteApiClient(
      baseUrl: baseUrl,
      timeout: timeout ?? const Duration(seconds: 30),
    );
    // 계층 분류(/predict-hier) 우선 — 대분류(항상 견고) + 세부(확신 시).
    // 구버전 서버(404) / 계층 모델 미배치(503) 는 기존 경로로 fallback.
    try {
      return await client.predictHier(image, onUploadProgress: onUploadProgress);
    } on ApiException catch (e) {
      if (e.statusCode == 404 || e.statusCode == 503) {
        appLog('[predict] hier 미지원 서버 (${e.statusCode}) → 기존 경로 fallback');
        return centered
            ? client.predictCentered(image, onUploadProgress: onUploadProgress)
            : client.predict(image, onUploadProgress: onUploadProgress);
      }
      rethrow;
    }
  }


  /// 피드백 전송 — 온디바이스 모드에선 인터넷 있을 때만 가능.
  /// 인터넷 없으면 silently fail (또는 UI 에서 경고).
  Future<FeedbackResult?> sendFeedback({
    required String uploadId,
    required bool confirmed,
    String? correctedLabel,
  }) async {
    final baseUrl = await _settings.getApiUrl();
    final client = WasteApiClient(baseUrl: baseUrl);
    try {
      return await client.sendFeedback(
        uploadId: uploadId,
        confirmed: confirmed,
        correctedLabel: correctedLabel,
      );
    } catch (_) {
      return null;  // 오프라인 등 — UI 에서 처리
    }
  }
}


/// modelArch 가 fallback 으로 반환됐는지 판별 (UI 배지용).
bool isCloudFallback(String modelArch) => modelArch.startsWith('cloud-fallback');

