import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/data/confidence.dart';
import 'package:greenguide/features/result/widgets/banners.dart';

import '../helpers/test_env.dart';

/// 계층 응답 — allProbabilities 는 대분류 확률(서버 /predict-hier 와 동일).
Prediction _hier(String level, Map<String, double> coarse) {
  final top = coarse.entries.reduce((a, b) => a.value >= b.value ? a : b);
  return Prediction(
    predictedClass: top.key,
    predictedIndex: 0,
    confidence: top.value,
    allProbabilities: coarse,
    modelArch: 'test',
    inferenceMs: 1,
    hier: HierInfo(
      displayLevel: level,
      coarseClass: top.key,
      coarseConfidence: top.value,
      fineConfidence: 0,
      fineMargin: 0,
    ),
  );
}

/// keepAll 이 끼운 단어 결합자(U+2060)를 벗기고 비교.
Finder _textHas(String needle) => find.byWidgetPredicate(
  (w) => w is Text && (w.data ?? '').replaceAll('\u2060', '').contains(needle),
);

void main() {
  group('isElectronicsPlasticConfusion', () {
    test('coarse + 상위 2개가 전자제품·플라스틱 → true (순서 무관)', () {
      expect(
        isElectronicsPlasticConfusion(
          _hier('coarse', {'electronics': 0.5, 'plastic': 0.4, 'paper': 0.1}),
        ),
        isTrue,
      );
      expect(
        isElectronicsPlasticConfusion(
          _hier('coarse', {'plastic': 0.6, 'electronics': 0.3, 'glass': 0.1}),
        ),
        isTrue,
      );
    });

    test('fine 까지 확신했거나 상위 쌍이 다르면 false', () {
      expect(
        isElectronicsPlasticConfusion(
          _hier('fine', {'electronics': 0.5, 'plastic': 0.4, 'paper': 0.1}),
        ),
        isFalse,
      );
      expect(
        isElectronicsPlasticConfusion(
          _hier('coarse', {'electronics': 0.5, 'paper': 0.4, 'plastic': 0.1}),
        ),
        isFalse,
      );
    });
  });

  group('재촬영 안내 문구', () {
    testWidgets('단일 reject 카드에 "틀 안에 가깝게" 안내', (tester) async {
      await tester.pumpWidget(
        wrapApp(
          Scaffold(
            body: SingleChildScrollView(
              child: RejectCard(
                prediction: _hier('reject', {
                  'electronics': 0.4,
                  'plastic': 0.35,
                  'etc': 0.25,
                }),
              ),
            ),
          ),
        ),
      );
      expect(_textHas('틀 안에 가깝게'), findsOneWidget);
    });

    testWidgets('다중 물체 reject 카드엔 재촬영 안내 없음', (tester) async {
      await tester.pumpWidget(
        wrapApp(
          Scaffold(
            body: SingleChildScrollView(
              child: RejectCard(
                prediction: _hier('reject', {
                  'electronics': 0.4,
                  'plastic': 0.35,
                  'etc': 0.25,
                }),
                isMultiObject: true,
              ),
            ),
          ),
        ),
      );
      expect(_textHas('틀 안에 가깝게'), findsNothing);
    });

    testWidgets('RetakeCloserTip — 전자제품·플라스틱 혼동 문구', (tester) async {
      await tester.pumpWidget(wrapApp(const Scaffold(body: RetakeCloserTip())));
      expect(_textHas('전자제품과 플라스틱'), findsOneWidget);
      expect(_textHas('틀 안에 가깝게'), findsOneWidget);
    });
  });
}
