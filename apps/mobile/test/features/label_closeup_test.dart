import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/features/result/widgets/banners.dart';

import '../helpers/test_env.dart';

/// 11-11 실사진 40e4a32a — 페트병 라벨 근접: CNN 종이, AI 정밀 분석 플라스틱.
Map<String, dynamic> _closeupJson({bool withNew = true}) => {
  'display_level': 'fine',
  'display_class': 'cardboard',
  'coarse_class': 'paper',
  'coarse_confidence': 0.84,
  'fine_class': 'cardboard',
  'fine_confidence': 0.83,
  'fine_margin': 0.5,
  'coarse_probabilities': {'paper': 0.84, 'vinyl': 0.04, 'trash': 0.02},
  'model_arch': 'test',
  'inference_ms': 10,
  if (withNew) ...{
    'needs_confirm': true,
    'alt_candidate': {
      'class': 'plastic_other',
      'coarse_class': 'plastic',
      'confidence': 0.6,
      'reason': '핸드 트리트먼트 화장품 용기로 보여요',
    },
    'retake_hint': 'whole_object',
  },
};

Finder _textHas(String needle) => find.byWidgetPredicate(
  (w) => w is Text && (w.data ?? '').replaceAll('⁠', '').contains(needle),
);

void main() {
  group('needs_confirm·alt_candidate·retake_hint 파싱', () {
    test('세 필드를 읽는다', () {
      final p = Prediction.fromHierJson(_closeupJson());
      expect(p.needsConfirm, isTrue);
      expect(p.altCandidate?.slug, 'plastic_other');
      expect(p.altCandidate?.coarseClass, 'plastic');
      expect(p.altCandidate?.confidence, 0.6);
      expect(p.retakeHint, 'whole_object');
    });

    test('구버전 서버(필드 없음)는 기본값', () {
      final p = Prediction.fromHierJson(_closeupJson(withNew: false));
      expect(p.needsConfirm, isFalse);
      expect(p.altCandidate, isNull);
      expect(p.retakeHint, isNull);
    });
  });

  group('배너', () {
    testWidgets('ConfirmBanner — 두 후보와 AI 정밀 분석 이유', (tester) async {
      final p = Prediction.fromHierJson(_closeupJson());
      await tester.pumpWidget(
        wrapApp(Scaffold(body: ConfirmBanner(prediction: p))),
      );
      expect(_textHas('한 번 확인해 주세요'), findsOneWidget);
      expect(_textHas('중 하나예요'), findsOneWidget);
      expect(_textHas('화장품 용기'), findsOneWidget);
    });

    testWidgets('RetakeWholeTip — 전체·표시 안내', (tester) async {
      await tester.pumpWidget(wrapApp(const Scaffold(body: RetakeWholeTip())));
      expect(_textHas('물건 전체와 분리배출 표시'), findsOneWidget);
    });
  });
}
