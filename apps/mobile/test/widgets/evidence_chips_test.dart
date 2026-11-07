import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/features/result/widgets/evidence_chips.dart';

void main() {
  testWidgets('긴 AI 정밀 분석 문장도 좁은 폭에서 넘치지 않고 줄바꿈된다', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 300,
            child: EvidenceChips(
              evidence: [
                EvidenceItem(
                  type: 'vlm',
                  token: '하이트진로 블랙보리 보리차의 무색 투명 PET 음료병입니다. '
                      '라벨이 몸통을 감싸고 있어 제거가 필요합니다.',
                  mappedClass: 'pet',
                  score: 0.9,
                ),
              ],
            ),
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    final text = find.textContaining('AI 정밀 분석');
    expect(text, findsOneWidget);
    expect(tester.getSize(text).width, lessThanOrEqualTo(300));
    expect(tester.getSize(text).height, greaterThan(20)); // 2줄 이상
  });
}
