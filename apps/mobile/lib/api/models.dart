/// waste-api 응답 모델.
/// Python 의 PredictionResponse / Pydantic 스키마와 1:1 대응.
library;

import 'dart:ui' show Color;

/// `/predict-hier` 의 계층 정보 — 대분류(항상) + 세부(게이트 통과 시).
class HierInfo {
  /// "fine" | "coarse" | "reject"
  final String displayLevel;
  final String coarseClass;
  final double coarseConfidence;

  /// 세부 게이트 통과 시에만 non-null.
  final String? fineClass;
  final double fineConfidence;
  final double fineMargin;

  const HierInfo({
    required this.displayLevel,
    required this.coarseClass,
    required this.coarseConfidence,
    required this.fineConfidence,
    required this.fineMargin,
    this.fineClass,
  });

  bool get isFine => displayLevel == 'fine' && fineClass != null;
  bool get isReject => displayLevel == 'reject';
}

/// 지역별 배출 규정 한 건 — /region-info 응답 (공공데이터 표준 필드).
class RegionRule {
  final String district; // 관리구역 (없으면 '')
  final String? methodGeneral; // 생활쓰레기 배출방법
  final String? methodFood; // 음식물 배출방법
  final String? methodRecycle; // 재활용품 배출방법
  final String? methodBulk; // 대형/일시다량 배출방법
  final String? daysGeneral;
  final String? daysFood;
  final String? daysRecycle;
  final String? emitTime;
  final String? noCollectDay;
  final String? phone;

  const RegionRule({
    required this.district,
    this.methodGeneral,
    this.methodFood,
    this.methodRecycle,
    this.methodBulk,
    this.daysGeneral,
    this.daysFood,
    this.daysRecycle,
    this.emitTime,
    this.noCollectDay,
    this.phone,
  });

  factory RegionRule.fromJson(Map<String, dynamic> json) => RegionRule(
    district: json['district'] as String? ?? '',
    methodGeneral: json['method_general'] as String?,
    methodFood: json['method_food'] as String?,
    methodRecycle: json['method_recycle'] as String?,
    methodBulk: json['method_bulk'] as String?,
    daysGeneral: json['days_general'] as String?,
    daysFood: json['days_food'] as String?,
    daysRecycle: json['days_recycle'] as String?,
    emitTime: json['emit_time'] as String?,
    noCollectDay: json['no_collect_day'] as String?,
    phone: json['phone'] as String?,
  );
}

/// /region-info 응답 — 선택 지역의 배출 규정 목록.
class RegionInfo {
  final String sido;
  final String sigungu;
  final List<RegionRule> rules;

  const RegionInfo({
    required this.sido,
    required this.sigungu,
    required this.rules,
  });

  factory RegionInfo.fromJson(Map<String, dynamic> json) => RegionInfo(
    sido: json['sido'] as String,
    sigungu: json['sigungu'] as String,
    rules: (json['rules'] as List<dynamic>? ?? const [])
        .map((r) => RegionRule.fromJson(r as Map<String, dynamic>))
        .toList(),
  );

  /// 대표 규정 — 시군구 공통(district='') 우선, 없으면 첫 항목.
  RegionRule? get representative {
    if (rules.isEmpty) return null;
    return rules.firstWhere(
      (r) => r.district.isEmpty,
      orElse: () => rules.first,
    );
  }
}

/// 시맨틱 증거 — 서버가 분류에 융합한 단서 (분리배출 마크·라벨 문구·형태 인식).
/// 결과 카드의 "왜 이렇게 분류했어?" 신뢰 배지에 사용.
class EvidenceItem {
  final String type; // "mark" | "text" | "identity"
  final String token; // 표시용 (예: 무색페트, 키보드)
  final String mappedClass;
  final double score;

  const EvidenceItem({
    required this.type,
    required this.token,
    required this.mappedClass,
    required this.score,
  });

  factory EvidenceItem.fromJson(Map<String, dynamic> json) => EvidenceItem(
    type: json['type'] as String,
    token: json['token'] as String,
    mappedClass: json['mapped_class'] as String,
    score: (json['score'] as num).toDouble(),
  );
}

class Prediction {
  final String predictedClass;
  final int predictedIndex;
  final double confidence;
  final Map<String, double> allProbabilities;
  final String modelArch;
  final double inferenceMs;
  final String? uploadId;

  /// 계층 분류(/predict-hier) 사용 시에만 non-null. 구버전 서버는 null.
  final HierInfo? hier;

  /// 서버가 융합한 시맨틱 증거 (없으면 빈 목록).
  final List<EvidenceItem> evidence;

  const Prediction({
    required this.predictedClass,
    required this.predictedIndex,
    required this.confidence,
    required this.allProbabilities,
    required this.modelArch,
    required this.inferenceMs,
    this.uploadId,
    this.hier,
    this.evidence = const [],
  });

  factory Prediction.fromJson(Map<String, dynamic> json) {
    final probs = (json['all_probabilities'] as Map<String, dynamic>).map(
      (k, v) => MapEntry(k, (v as num).toDouble()),
    );
    return Prediction(
      predictedClass: json['predicted_class'] as String,
      predictedIndex: json['predicted_index'] as int,
      confidence: (json['confidence'] as num).toDouble(),
      allProbabilities: probs,
      modelArch: json['model_arch'] as String,
      inferenceMs: (json['inference_ms'] as num).toDouble(),
      uploadId: json['upload_id'] as String?,
    );
  }

  /// `/predict-hier` 응답 → Prediction (기존 UI 와 호환되는 형태로 매핑).
  ///
  /// - predictedClass = display_class (게이트 적용된 노출 slug)
  /// - allProbabilities = 대분류 확률 분포 (확률 카드용)
  /// - confidence = 노출 레벨에 맞는 확신도
  factory Prediction.fromHierJson(Map<String, dynamic> json) {
    final coarseProbs = (json['coarse_probabilities'] as Map<String, dynamic>)
        .map((k, v) => MapEntry(k, (v as num).toDouble()));
    final level = json['display_level'] as String;
    final hier = HierInfo(
      displayLevel: level,
      coarseClass: json['coarse_class'] as String,
      coarseConfidence: (json['coarse_confidence'] as num).toDouble(),
      fineClass: json['fine_class'] as String?,
      fineConfidence: (json['fine_confidence'] as num).toDouble(),
      fineMargin: (json['fine_margin'] as num).toDouble(),
    );
    return Prediction(
      predictedClass: json['display_class'] as String,
      predictedIndex: -1,
      confidence: level == 'fine' ? hier.fineConfidence : hier.coarseConfidence,
      allProbabilities: coarseProbs,
      modelArch: json['model_arch'] as String,
      inferenceMs: (json['inference_ms'] as num).toDouble(),
      uploadId: json['upload_id'] as String?,
      hier: hier,
      evidence: (json['evidence'] as List<dynamic>? ?? const [])
          .map((e) => EvidenceItem.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }

  /// 확률 내림차순으로 정렬된 (라벨, 확률) 리스트.
  List<MapEntry<String, double>> sortedProbabilities() {
    final entries = allProbabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return entries;
  }
}

class FeedbackResult {
  final String uploadId;
  final String feedbackStatus;
  final String feedbackLabel;

  const FeedbackResult({
    required this.uploadId,
    required this.feedbackStatus,
    required this.feedbackLabel,
  });

  factory FeedbackResult.fromJson(Map<String, dynamic> json) {
    return FeedbackResult(
      uploadId: json['upload_id'] as String,
      feedbackStatus: json['feedback_status'] as String,
      feedbackLabel: json['feedback_label'] as String,
    );
  }
}

class ServiceInfo {
  final String name;
  final String version;
  final String modelArch;
  final String modelPath;
  final List<String> classLabels;
  final int maxUploadSizeBytes;

  const ServiceInfo({
    required this.name,
    required this.version,
    required this.modelArch,
    required this.modelPath,
    required this.classLabels,
    required this.maxUploadSizeBytes,
  });

  factory ServiceInfo.fromJson(Map<String, dynamic> json) {
    return ServiceInfo(
      name: json['name'] as String,
      version: json['version'] as String,
      modelArch: json['model_arch'] as String,
      modelPath: json['model_path'] as String,
      classLabels: List<String>.from(json['class_labels'] as List),
      maxUploadSizeBytes: json['max_upload_size_bytes'] as int,
    );
  }
}

/// `/predict-with-cam` 응답 — 예측 + heatmap PNG (base64 data URI).
class PredictionWithCam {
  final Prediction prediction;

  /// `data:image/png;base64,...` 형식 (Flutter `Image.memory` 로 디코드해서 표시).
  final String? camBase64;

  /// false 면 서버가 cam-aware ONNX 가 아니라 CAM 생성 불가.
  final bool camAvailable;

  const PredictionWithCam({
    required this.prediction,
    required this.camAvailable,
    this.camBase64,
  });

  factory PredictionWithCam.fromJson(Map<String, dynamic> json) {
    return PredictionWithCam(
      prediction: Prediction.fromJson(json),
      camBase64: json['cam_base64'] as String?,
      camAvailable: json['cam_available'] as bool? ?? false,
    );
  }

  /// `/predict-hier?want_cam=true` 응답 — 결과 카드를 만든 것과 같은 요청의 CAM.
  /// 구버전 서버는 want_cam 을 무시하고 cam_base64 를 내려주지 않으므로 그 경우 미지원.
  factory PredictionWithCam.fromHierJson(Map<String, dynamic> json) {
    final cam = json['cam_base64'] as String?;
    return PredictionWithCam(
      prediction: Prediction.fromHierJson(json),
      camBase64: cam,
      camAvailable: cam != null,
    );
  }
}

/// `/predict-with-regions` 에서 검출된 한 재질 영역.
class MaterialRegion {
  final String slug;

  /// [x0,y0,x1,y1] 0~1 (라벨 위치용).
  final List<double> bboxNorm;
  final double avgConf;
  final int cellCount;

  /// 서버가 빗금 오버레이에 쓴 색 (#RRGGBB). 배지·목록이 같은 색을 쓰기 위함.
  final String? colorHex;

  const MaterialRegion({
    required this.slug,
    required this.bboxNorm,
    required this.avgConf,
    required this.cellCount,
    this.colorHex,
  });

  /// [colorHex] 를 파싱한 색. 없거나 형식이 다르면 null (앱 팔레트로 폴백).
  Color? get color {
    final hex = colorHex;
    if (hex == null) return null;
    final m = RegExp(r'^#?([0-9a-fA-F]{6})$').firstMatch(hex);
    if (m == null) return null;
    return Color(0xFF000000 | int.parse(m.group(1)!, radix: 16));
  }

  factory MaterialRegion.fromJson(Map<String, dynamic> json) {
    return MaterialRegion(
      slug: json['slug'] as String,
      bboxNorm: (json['bbox_norm'] as List)
          .map((e) => (e as num).toDouble())
          .toList(),
      avgConf: (json['avg_conf'] as num).toDouble(),
      cellCount: json['cell_count'] as int,
      colorHex: json['color_hex'] as String?,
    );
  }

  /// bbox 중심 (정규화 0~1).
  double get cx => (bboxNorm[0] + bboxNorm[2]) / 2;
  double get cy => (bboxNorm[1] + bboxNorm[3]) / 2;
}

/// `/predict-with-regions` 응답 — 예측 + 다중재질 영역 + 원본 위 빗금 오버레이.
class PredictionWithRegions {
  final Prediction prediction;

  /// 원본에 영역별 빗금을 그린 JPEG (data:image/jpeg;base64,…). 누끼 대신 표시.
  final String? overlayBase64;

  /// 검출된 재질 영역들 (확실히 다른 재질만). 1개면 단일재질, 2+면 다중재질.
  final List<MaterialRegion> regions;
  final int gridH;
  final int gridW;

  const PredictionWithRegions({
    required this.prediction,
    required this.regions,
    this.overlayBase64,
    this.gridH = 0,
    this.gridW = 0,
  });

  factory PredictionWithRegions.fromJson(Map<String, dynamic> json) {
    return PredictionWithRegions(
      prediction: Prediction.fromJson(json),
      overlayBase64: json['overlay_base64'] as String?,
      regions: ((json['regions'] as List?) ?? const [])
          .map((e) => MaterialRegion.fromJson(e as Map<String, dynamic>))
          .toList(),
      gridH: json['grid_h'] as int? ?? 0,
      gridW: json['grid_w'] as int? ?? 0,
    );
  }

  /// 2개 이상 재질이 검출됐는지 (다중재질).
  bool get isMultiMaterial => regions.length >= 2;
  bool get hasOverlay => overlayBase64 != null && regions.isNotEmpty;
}

/// `/predict-objects` 의 객체 후보 — 혼재 장면에서 분리된 물건 하나.
class ObjectCandidate {
  final List<double> bboxNorm;
  final String displayLevel;
  final String displayClass;
  final String coarseClass;
  final double coarseConfidence;
  final String? fineClass;
  final double fineConfidence;
  final Map<String, double> coarseProbabilities;

  const ObjectCandidate({
    required this.bboxNorm,
    required this.displayLevel,
    required this.displayClass,
    required this.coarseClass,
    required this.coarseConfidence,
    required this.fineConfidence,
    required this.coarseProbabilities,
    this.fineClass,
  });

  factory ObjectCandidate.fromJson(Map<String, dynamic> json) {
    return ObjectCandidate(
      bboxNorm: (json['bbox_norm'] as List)
          .map((e) => (e as num).toDouble())
          .toList(),
      displayLevel: json['display_level'] as String,
      displayClass: json['display_class'] as String,
      coarseClass: json['coarse_class'] as String,
      coarseConfidence: (json['coarse_confidence'] as num).toDouble(),
      fineClass: json['fine_class'] as String?,
      fineConfidence: (json['fine_confidence'] as num?)?.toDouble() ?? 0.0,
      coarseProbabilities:
          ((json['coarse_probabilities'] as Map<String, dynamic>?) ?? {}).map(
            (k, v) => MapEntry(k, (v as num).toDouble()),
          ),
    );
  }

  double get cx => (bboxNorm[0] + bboxNorm[2]) / 2;
  double get cy => (bboxNorm[1] + bboxNorm[3]) / 2;

  /// 후보 선택 시 메인 결과 카드로 쓸 Prediction 으로 변환.
  Prediction toPrediction() {
    return Prediction(
      predictedClass: displayClass,
      predictedIndex: -1,
      confidence: displayLevel == 'fine' ? fineConfidence : coarseConfidence,
      allProbabilities: coarseProbabilities,
      modelArch: 'object-select',
      inferenceMs: 0,
      hier: HierInfo(
        displayLevel: displayLevel,
        coarseClass: coarseClass,
        coarseConfidence: coarseConfidence,
        fineClass: fineClass,
        fineConfidence: fineConfidence,
        fineMargin: 0,
      ),
    );
  }
}

/// `/predict-objects` 응답.
class PredictObjects {
  final List<ObjectCandidate> objects;
  final double inferenceMs;

  const PredictObjects({required this.objects, required this.inferenceMs});

  factory PredictObjects.fromJson(Map<String, dynamic> json) {
    return PredictObjects(
      objects: ((json['objects'] as List?) ?? const [])
          .map((e) => ObjectCandidate.fromJson(e as Map<String, dynamic>))
          .toList(),
      inferenceMs: (json['inference_ms'] as num?)?.toDouble() ?? 0.0,
    );
  }

  bool get isMultiObject => objects.length >= 2;
}

/// 업로드 메타 — 학습/서빙 분포 정렬용 폼 필드 (필드명은 waste-api 와 합의됨).
///
/// - `capture_mode`: "smart" | "gallery"
/// - `orientation`: 원본 EXIF Orientation (1/3/6/8; 갤러리는 picker 가 회전을
///   픽셀에 반영한 뒤라 1 = 정보 없음)
/// - `quality_blur` / `quality_brightness`: 클라이언트 품질 측정값
/// - `crop_applied` / `crop_box`: 가이드 프레임 크롭 적용 여부·원본 좌표
class UploadMeta {
  final String captureMode;
  final int orientation;
  final double? qualityBlur;
  final double? qualityBrightness;
  final bool cropApplied;
  final String? cropBox; // "x,y,w,h" (정수, 원본 픽셀)

  const UploadMeta({
    required this.captureMode,
    this.orientation = 1,
    this.qualityBlur,
    this.qualityBrightness,
    this.cropApplied = false,
    this.cropBox,
  });

  Map<String, String> toFields() => {
    'capture_mode': captureMode,
    'orientation': '$orientation',
    if (qualityBlur != null) 'quality_blur': qualityBlur!.toStringAsFixed(2),
    if (qualityBrightness != null)
      'quality_brightness': qualityBrightness!.toStringAsFixed(2),
    if (cropApplied) 'crop_applied': 'true',
    if (cropApplied && cropBox != null) 'crop_box': cropBox!,
  };
}
