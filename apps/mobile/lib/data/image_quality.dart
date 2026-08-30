/// 캡처된 사진의 품질 평가 — 어둡거나 흔들린 사진은 분류 신뢰도가 낮으므로
/// 사용자에게 "다시 찍어주세요" 안내하기 위한 휴리스틱.
///
/// - 밝기: grayscale 평균 luminance (0-255). 너무 낮으면 어두움.
/// - 선명도: Laplacian variance. 낮으면 흔들림/초점 안 맞음.
///
/// 속도를 위해 256px 로 다운스케일 후 계산 (캡처당 1회, ~50-100ms).
library;

import 'dart:io';

import 'package:image/image.dart' as img;


enum ImageQualityIssue { tooDark, tooBlurry }


class ImageQualityResult {
  final double brightness;  // 0-255
  final double sharpness;   // laplacian variance
  final List<ImageQualityIssue> issues;

  const ImageQualityResult({
    required this.brightness,
    required this.sharpness,
    required this.issues,
  });

  bool get hasIssue => issues.isNotEmpty;
}


// 임계값 — 보수적으로 (false positive 최소화). 운영하며 튜닝.
const double _kDarkThreshold = 45.0;     // 평균 밝기 이 미만이면 어두움
const double _kBlurThreshold = 80.0;     // laplacian variance 이 미만이면 흐림
const int _kAnalysisSize = 256;          // 분석용 다운스케일 크기


Future<ImageQualityResult> assessImageQuality(File file) async {
  try {
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      return const ImageQualityResult(brightness: 128, sharpness: 999, issues: []);
    }

    final resized = img.copyResize(decoded, width: _kAnalysisSize);
    final gray = img.grayscale(resized);
    final w = gray.width;
    final h = gray.height;

    // 1) 평균 밝기
    double sum = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        sum += gray.getPixel(x, y).r.toDouble();
      }
    }
    final brightness = sum / (w * h);

    // 2) Laplacian variance (선명도)
    final sharpness = _laplacianVariance(gray);

    final issues = <ImageQualityIssue>[];
    if (brightness < _kDarkThreshold) issues.add(ImageQualityIssue.tooDark);
    if (sharpness < _kBlurThreshold) issues.add(ImageQualityIssue.tooBlurry);

    return ImageQualityResult(
      brightness: brightness,
      sharpness: sharpness,
      issues: issues,
    );
  } catch (_) {
    // 분석 실패는 품질 OK 로 간주 (분류 자체를 막지 않음)
    return const ImageQualityResult(brightness: 128, sharpness: 999, issues: []);
  }
}


/// 3×3 Laplacian kernel ([[0,1,0],[1,-4,1],[0,1,0]]) 의 응답 분산.
/// 선명한 이미지일수록 엣지 응답이 강해 분산이 큼.
double _laplacianVariance(img.Image gray) {
  final w = gray.width;
  final h = gray.height;
  final n = (w - 2) * (h - 2);
  if (n <= 0) return 999;

  // 1-pass mean, 2-pass variance (메모리 절약 위해 합/제곱합 누적)
  double sumLap = 0;
  double sumLapSq = 0;
  for (var y = 1; y < h - 1; y++) {
    for (var x = 1; x < w - 1; x++) {
      final c = gray.getPixel(x, y).r.toDouble();
      final up = gray.getPixel(x, y - 1).r.toDouble();
      final down = gray.getPixel(x, y + 1).r.toDouble();
      final left = gray.getPixel(x - 1, y).r.toDouble();
      final right = gray.getPixel(x + 1, y).r.toDouble();
      final lap = up + down + left + right - 4 * c;
      sumLap += lap;
      sumLapSq += lap * lap;
    }
  }
  final mean = sumLap / n;
  return (sumLapSq / n) - (mean * mean);
}
