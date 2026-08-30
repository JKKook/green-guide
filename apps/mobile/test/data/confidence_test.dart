import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/data/confidence.dart';

Prediction _pred(Map<String, double> probs) {
  final top = probs.entries.reduce((a, b) => a.value >= b.value ? a : b);
  return Prediction(
    predictedClass: top.key,
    predictedIndex: 0,
    confidence: top.value,
    allProbabilities: probs,
    modelArch: 'test',
    inferenceMs: 1,
  );
}

void main() {
  group('assessConfidence', () {
    test('top1 ≥ 0.80 이고 격차가 크면 high', () {
      final a = assessConfidence(
        _pred({'paper': 0.9, 'glass': 0.05, 'can': 0.05}),
      );
      expect(a.level, ConfidenceLevel.high);
      expect(a.isUncertain, isFalse);
      expect(a.shouldReject, isFalse);
    });

    test('0.60 ≤ top1 < 0.80 이면 medium', () {
      final a = assessConfidence(
        _pred({'paper': 0.7, 'glass': 0.2, 'can': 0.1}),
      );
      expect(a.level, ConfidenceLevel.medium);
    });

    test('top1·top2 격차가 0.15 미만이면 확률이 높아도 low', () {
      final a = assessConfidence(
        _pred({'paper': 0.48, 'cardboard': 0.45, 'can': 0.07}),
      );
      expect(a.level, ConfidenceLevel.low);
      expect(a.margin, closeTo(0.03, 1e-9));
    });

    test('top1 < 0.55 면 reject', () {
      final a = assessConfidence(
        _pred({'paper': 0.5, 'glass': 0.3, 'can': 0.2}),
      );
      expect(a.shouldReject, isTrue);
    });

    test('분포가 평평하면(entropy > 0.7) top1 이 기준을 넘어도 reject', () {
      // 8클래스 균등에 가까운 분포 — 정규화 엔트로피 ≈ 1
      final probs = {for (var i = 0; i < 8; i++) 'c$i': 0.125};
      final a = assessConfidence(_pred(probs));
      expect(a.normalizedEntropy, closeTo(1.0, 1e-6));
      expect(a.shouldReject, isTrue);
    });

    test('one-hot 분포의 정규화 엔트로피는 0', () {
      final a = assessConfidence(_pred({'paper': 1.0, 'glass': 0.0}));
      expect(a.normalizedEntropy, 0.0);
    });
  });

  test('레벨별 라벨', () {
    expect(confidenceLabel(ConfidenceLevel.high), '확신');
    expect(confidenceLabel(ConfidenceLevel.medium), '보통');
    expect(confidenceLabel(ConfidenceLevel.low), '불확실');
  });
}
