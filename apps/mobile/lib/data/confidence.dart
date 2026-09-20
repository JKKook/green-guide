/// 예측 신뢰도 평가 — 모델이 "확신하는지 / 애매한지" 판단.
///
/// 단순 top-1 confidence 뿐 아니라 top-1 과 top-2 의 격차(margin)도 본다.
/// 두 클래스가 비슷하게 나오면 (예: paper 0.45 / cardboard 0.40) confidence 가
/// 높아 보여도 사실 모델은 헷갈리는 상태 → 불확실로 처리.
library;

import 'dart:math';

import 'package:flutter/material.dart';

import '../api/models.dart';


enum ConfidenceLevel { high, medium, low }


class ConfidenceAssessment {
  final ConfidenceLevel level;
  final double topConfidence;       // top-1 확률
  final double margin;              // top-1 − top-2
  final double normalizedEntropy;   // 0(one-hot) ~ 1(uniform). 0.7+ 면 분포 매우 평평

  const ConfidenceAssessment({
    required this.level,
    required this.topConfidence,
    required this.margin,
    required this.normalizedEntropy,
  });

  bool get isUncertain => level == ConfidenceLevel.low;

  /// open-set reject — 모델이 어느 클래스에도 확신 못 함.
  /// 틀린 추측을 단정하느니 "기타/분류 불가" 로 결론짓는다.
  ///
  /// 2026-05-31 강화: top1 단독 임계 0.45→0.55 + entropy gate 추가.
  /// 가설 2 분석에서 클래스 가중치 amplification 으로 cardboard/non_object 가
  /// OOD sink 가 되어 over-fire 가 다수 — 더 엄격한 reject 가 필요.
  bool get shouldReject =>
      topConfidence < _kRejectThreshold ||
      normalizedEntropy > _kEntropyThreshold;
}


// 임계값 — 운영하며 튜닝
const double _kHighThreshold = 0.80;     // 이 이상이면 자신 있음 (녹색)
const double _kLowThreshold = 0.60;      // 이 미만이면 불확실 (빨강)
const double _kMarginThreshold = 0.15;   // top1·top2 격차가 이 미만이면 불확실
const double _kRejectThreshold = 0.55;   // top1 이 이 미만 → 분류 불가(이전 0.45 에서 강화)
const double _kEntropyThreshold = 0.70;  // softmax 분포가 평평하면(엔트로피 높음) reject


/// 정규화 entropy 계산 — 0(one-hot, 완전 확신) ~ 1(uniform, 완전 분산).
double _normalizedEntropy(List<double> probs) {
  if (probs.isEmpty) return 0.0;
  double h = 0.0;
  for (final p in probs) {
    if (p > 1e-12) h -= p * log(p);
  }
  final maxH = log(probs.length.toDouble());
  return maxH > 0 ? (h / maxH).clamp(0.0, 1.0) : 0.0;
}


ConfidenceAssessment assessConfidence(Prediction p) {
  final probs = p.allProbabilities.values.toList();
  final sorted = List<double>.from(probs)
    ..sort((a, b) => b.compareTo(a));
  final top1 = sorted.isNotEmpty ? sorted[0] : p.confidence;
  final top2 = sorted.length > 1 ? sorted[1] : 0.0;
  final margin = top1 - top2;
  final entropy = _normalizedEntropy(probs);

  final ConfidenceLevel level;
  if (top1 < _kLowThreshold || margin < _kMarginThreshold) {
    level = ConfidenceLevel.low;
  } else if (top1 < _kHighThreshold) {
    level = ConfidenceLevel.medium;
  } else {
    level = ConfidenceLevel.high;
  }

  return ConfidenceAssessment(
    level: level,
    topConfidence: top1,
    margin: margin,
    normalizedEntropy: entropy,
  );
}


/// 신뢰도 레벨 → 색상 (녹/노/빨).
Color confidenceColor(ConfidenceLevel level) {
  switch (level) {
    case ConfidenceLevel.high:
      return const Color(0xFF2E7D32);  // green
    case ConfidenceLevel.medium:
      return const Color(0xFFF9A825);  // amber
    case ConfidenceLevel.low:
      return const Color(0xFFD32F2F);  // red
  }
}


/// 신뢰도 레벨 → 짧은 한글 라벨.
String confidenceLabel(ConfidenceLevel level) {
  switch (level) {
    case ConfidenceLevel.high:
      return '확신';
    case ConfidenceLevel.medium:
      return '보통';
    case ConfidenceLevel.low:
      return '불확실';
  }
}


/// 결과 화면의 "분류 불가(reject)" 판정 — 셋 중 하나면 reject:
/// (1) 신뢰도 부족 (top1 < 0.55 또는 정규화 엔트로피 > 0.7)
/// (2) 모델이 명시적으로 non_object (폐기물 아님 — 재촬영 신호)
/// (3) 계층 응답이 reject (대분류조차 불확실)
/// 결과 카드와 사진 위 영역 배지가 같은 규칙을 쓰도록 한 곳에 둔다.
bool isRejectPrediction(Prediction p) =>
    assessConfidence(p).shouldReject ||
    p.predictedClass == 'non_object' ||
    (p.hier?.isReject ?? false);
