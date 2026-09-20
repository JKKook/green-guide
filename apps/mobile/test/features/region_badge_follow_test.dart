import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/features/result/widgets/regions_view.dart';

Prediction _pred(String cls, {double conf = 0.9}) => Prediction(
  predictedClass: cls,
  predictedIndex: 0,
  confidence: conf,
  allProbabilities: {cls: conf, 'etc': 1 - conf},
  modelArch: 'test',
  inferenceMs: 1,
);

PredictionWithRegions _regions(List<String> slugs) =>
    PredictionWithRegions.fromJson({
      'predicted_class': slugs.first,
      'predicted_index': 0,
      'confidence': 0.9,
      'all_probabilities': {slugs.first: 0.9},
      'model_arch': 'test',
      'inference_ms': 1,
      'overlay_base64': 'data:image/jpeg;base64,AAAA',
      'regions': [
        for (final s in slugs)
          {
            'slug': s,
            'bbox_norm': [0, 0, 1, 1],
            'avg_conf': 0.8,
            'cell_count': 1,
          },
      ],
    });

void main() {
  test('영역 하나 + 카드가 확신하면 배지는 카드 재질을 따른다', () {
    final info = badgeOverrideFor(_regions(['electronics']), _pred('plastic'));
    expect(info?.classKey, 'plastic');
  });

  test('영역 2개 이상이면 영역별 라벨 유지', () {
    expect(
      badgeOverrideFor(_regions(['electronics', 'plastic']), _pred('plastic')),
      isNull,
    );
  });

  test('카드가 reject(저확신)면 영역 라벨 유지 — 영역 승격 분기', () {
    expect(
      badgeOverrideFor(_regions(['electronics']), _pred('plastic', conf: 0.4)),
      isNull,
    );
    expect(badgeOverrideFor(_regions(['electronics']), null), isNull);
  });
}
