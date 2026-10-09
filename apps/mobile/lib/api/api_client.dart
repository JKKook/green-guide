/// waste-api 호출 클라이언트.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

import 'models.dart';

class ApiException implements Exception {
  final int? statusCode;
  final String message;
  ApiException(this.message, {this.statusCode});

  @override
  String toString() => 'ApiException(${statusCode ?? '-'}): $message';
}

/// 사용자에게 보여줄 한국어 오류 문구.
///
/// `TimeoutException after 0:00:30...` 같은 원시 예외 문자열이 결과 화면·
/// 스낵바에 그대로 노출되던 것을 막는다. 절전 중인 서버(HF Spaces 무료 티어)가
/// 깨어나는 상황이 가장 흔한 실패라 그 안내를 우선한다.
String friendlyError(Object e) {
  if (e is TimeoutException) {
    return '서버 응답이 늦어지고 있어요.\n'
        '절전 중인 서버가 깨어나는 중일 수 있어요 — 잠시 후 다시 시도해 주세요.';
  }
  if (e is SocketException || e is http.ClientException) {
    return '서버에 연결할 수 없어요.\n인터넷 연결을 확인한 뒤 다시 시도해 주세요.';
  }
  if (e is ApiException) {
    return switch (e.statusCode) {
      400 => '사진을 분석할 수 없었어요. 다른 각도로 다시 촬영해 주세요.',
      404 => '서버에서 이 기능을 아직 지원하지 않아요.',
      413 => '사진 용량이 너무 커요. 다시 촬영해 주세요.',
      429 => '요청이 많아요. 잠시 후 다시 시도해 주세요.',
      503 => '서버가 준비 중이에요. 잠시 후 다시 시도해 주세요.',
      _ => '서버에서 문제가 생겼어요. 잠시 후 다시 시도해 주세요.',
    };
  }
  return '분석에 실패했어요. 잠시 후 다시 시도해 주세요.';
}

class WasteApiClient {
  String baseUrl;
  final Duration timeout;

  WasteApiClient({
    required this.baseUrl,
    this.timeout = const Duration(seconds: 30),
  });

  Uri _uri(String path) {
    final base = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return Uri.parse('$base$path');
  }

  Future<bool> isHealthy() async {
    try {
      final res = await http.get(_uri('/health')).timeout(timeout);
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// 지역별 배출 규정 (공공데이터 기반). 미적재/실패 시 null — 전국 공통 안내 fallback.
  Future<RegionInfo?> fetchRegionInfo(String sido, String sigungu) async {
    try {
      final res = await http
          .get(
            _uri(
              '/region-info?sido=${Uri.encodeComponent(sido)}'
              '&sigungu=${Uri.encodeComponent(sigungu)}',
            ),
          )
          .timeout(timeout);
      if (res.statusCode != 200) return null;
      final json =
          jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      if ((json['count'] as int? ?? 0) == 0) return null;
      return RegionInfo.fromJson(json);
    } catch (_) {
      return null;
    }
  }

  Future<ServiceInfo> fetchServiceInfo() async {
    final res = await http.get(_uri('/')).timeout(timeout);
    if (res.statusCode != 200) {
      throw ApiException('서비스 정보 조회 실패', statusCode: res.statusCode);
    }
    return ServiceInfo.fromJson(
      jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>,
    );
  }

  Future<Prediction> predict(
    File imageFile, {
    UploadMeta? meta,
    UploadProgress? onUploadProgress,
  }) async {
    final json = await _multipartPostJson(
      '/predict',
      imageFile,
      onUploadProgress: onUploadProgress,
      fields: meta?.toFields(),
    );
    return Prediction.fromJson(json);
  }

  /// `/predict-hier` — 계층 분류: 대분류(항상) + 세부(신뢰도 게이트 통과 시).
  /// 구버전 서버(엔드포인트 없음)는 404/503 → 호출부에서 predict() fallback.
  ///
  /// [tapX]/[tapY] (정규화 0~1): 탭-투-셀렉트 — 혼재 장면에서 사용자가
  /// 지목한 객체의 saliency 성분만 서버가 크롭해 분류.
  Future<Prediction> predictHier(
    File imageFile, {
    double? tapX,
    double? tapY,
    UploadMeta? meta,
    UploadProgress? onUploadProgress,
  }) async {
    final json = await _multipartPostJson(
      '/predict-hier',
      imageFile,
      timeoutOverride: const Duration(seconds: 30),
      onUploadProgress: onUploadProgress,
      fields: {
        ...?meta?.toFields(),
        if (tapX != null && tapY != null) ...{
          'tap_x': tapX.toStringAsFixed(4),
          'tap_y': tapY.toStringAsFixed(4),
        },
      },
    );
    return Prediction.fromHierJson(json);
  }

  /// `/predict-centered` — u2netp 자동 객체 크롭 → 분류.
  /// Smart capture 가 사용해 객체 중심 입력으로 분류 정확도 ↑
  /// (Test C1 70% 크롭 +4.4pp 효과 직접 적용).
  Future<Prediction> predictCentered(
    File imageFile, {
    UploadMeta? meta,
    UploadProgress? onUploadProgress,
  }) async {
    final json = await _multipartPostJson(
      '/predict-centered',
      imageFile,
      timeoutOverride: const Duration(seconds: 30), // u2netp 분리 추가 시간 여유
      onUploadProgress: onUploadProgress,
      fields: meta?.toFields(),
    );
    return Prediction.fromJson(json);
  }

  /// `/predict-objects` — 혼재 장면의 객체 후보들 (탐지-후-분류).
  /// 각 saliency 성분을 개별 계층 분류. 후보 ≥2 면 다중 물건 장면.
  Future<PredictObjects> predictObjects(File imageFile) async {
    final json = await _multipartPostJson(
      '/predict-objects',
      imageFile,
      timeoutOverride: const Duration(seconds: 30),
    );
    return PredictObjects.fromJson(json);
  }

  /// `/predict-with-cam` — 예측 + heatmap PNG (base64 data URI).
  /// 서버가 cam-aware ONNX 가 아니면 `camAvailable=false` + `camBase64=null`.
  Future<PredictionWithCam> predictWithCam(File imageFile) async {
    // CAM 렌더링이 추가되어 약간 더 오래 걸릴 수 있음 — timeout 여유 두기
    final json = await _multipartPostJson(
      '/predict-with-cam',
      imageFile,
      timeoutOverride: const Duration(seconds: 30),
    );
    return PredictionWithCam.fromJson(json);
  }

  /// `/predict-hier` + `want_cam=true` — 결과 카드와 같은 모델·탭 크롭·prior 로
  /// 만든 CAM. `/predict-with-cam`(구형 단일 분류기·전체 프레임) 과 달리 표시
  /// 결과와 히트맵이 어긋나지 않는다. 탭 좌표는 결과를 만들 때 쓴 값을 그대로.
  Future<PredictionWithCam> predictHierCam(
    File imageFile, {
    double? tapX,
    double? tapY,
  }) async {
    final json = await _multipartPostJson(
      '/predict-hier',
      imageFile,
      timeoutOverride: const Duration(seconds: 30),
      fields: {
        'want_cam': 'true',
        if (tapX != null && tapY != null) ...{
          'tap_x': tapX.toStringAsFixed(4),
          'tap_y': tapY.toStringAsFixed(4),
        },
      },
    );
    return PredictionWithCam.fromHierJson(json);
  }

  /// `/predict-with-regions` — 예측 + 다중재질 영역 + 원본 위 빗금 오버레이.
  /// 확실히 다른 재질만 영역으로 분리 (없으면 1개 = 단일재질).
  Future<PredictionWithRegions> predictWithRegions(
    File imageFile, {
    double? tapX,
    double? tapY,
  }) async {
    final json = await _multipartPostJson(
      '/predict-with-regions',
      imageFile,
      timeoutOverride: const Duration(seconds: 30),
      fields: {
        if (tapX != null) 'tap_x': tapX.toStringAsFixed(4),
        if (tapY != null) 'tap_y': tapY.toStringAsFixed(4),
      },
    );
    return PredictionWithRegions.fromJson(json);
  }

  Future<Map<String, dynamic>> _multipartPostJson(
    String path,
    File imageFile, {
    Duration? timeoutOverride,
    Map<String, String>? fields,
    UploadProgress? onUploadProgress,
  }) async {
    final request = _ProgressMultipartRequest(
      'POST',
      _uri(path),
      onProgress: onUploadProgress,
    );
    request.files.add(
      await http.MultipartFile.fromPath(
        'image',
        imageFile.path,
        contentType: _guessMediaType(imageFile.path),
      ),
    );
    if (fields != null) request.fields.addAll(fields);
    final streamedRes = await request.send().timeout(
      timeoutOverride ?? timeout,
    );
    final res = await http.Response.fromStream(streamedRes);
    if (res.statusCode != 200) {
      String detail = '';
      try {
        final body =
            jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
        detail = body['detail']?.toString() ?? body.toString();
      } catch (_) {
        detail = utf8.decode(res.bodyBytes);
      }
      throw ApiException('$path 호출 실패: $detail', statusCode: res.statusCode);
    }
    return jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
  }

  Future<FeedbackResult> sendFeedback({
    required String uploadId,
    required bool confirmed,
    String? correctedLabel,
  }) async {
    final body = jsonEncode({
      'upload_id': uploadId,
      'confirmed': confirmed,
      'corrected_label': ?correctedLabel,
    });
    final res = await http
        .post(
          _uri('/feedback'),
          headers: {'Content-Type': 'application/json'},
          body: body,
        )
        .timeout(timeout);

    if (res.statusCode != 200) {
      String detail = '';
      try {
        final body =
            jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
        detail = body['detail']?.toString() ?? body.toString();
      } catch (_) {
        detail = utf8.decode(res.bodyBytes);
      }
      throw ApiException('피드백 전송 실패: $detail', statusCode: res.statusCode);
    }

    return FeedbackResult.fromJson(
      jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>,
    );
  }

  static http.MediaType? _guessMediaType(String path) {
    final lower = path.toLowerCase();
    if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
      return http.MediaType('image', 'jpeg');
    }
    if (lower.endsWith('.png')) return http.MediaType('image', 'png');
    if (lower.endsWith('.webp')) return http.MediaType('image', 'webp');
    return null;
  }
}

/// 업로드 진행률 콜백 — (보낸 바이트, 전체 바이트).
typedef UploadProgress = void Function(int sent, int total);

/// multipart 바디 스트림을 감싸 실제 전송 바이트를 세는 요청.
class _ProgressMultipartRequest extends http.MultipartRequest {
  _ProgressMultipartRequest(super.method, super.url, {this.onProgress});

  final UploadProgress? onProgress;

  @override
  http.ByteStream finalize() {
    final byteStream = super.finalize();
    final cb = onProgress;
    if (cb == null) return byteStream;
    final total = contentLength;
    var sent = 0;
    final transformer = StreamTransformer<List<int>, List<int>>.fromHandlers(
      handleData: (data, sink) {
        sent += data.length;
        cb(sent, total);
        sink.add(data);
      },
    );
    return http.ByteStream(byteStream.transform(transformer));
  }
}
