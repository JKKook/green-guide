/// 서드파티 고지 — 설정 > 앱 정보 > "오픈소스 라이선스" 목록에 추가된다.
///
/// 번들 폰트(OFL)와 AI 모델 학습 데이터 출처는 Flutter 가 자동 수집하는
/// 패키지 라이선스에 포함되지 않으므로 여기서 직접 등록한다.
library;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;

/// 앱 시작 시 1회 호출.
void registerBundledLicenses() {
  LicenseRegistry.addLicense(() async* {
    yield LicenseEntryWithLineBreaks(
      const ['Pretendard'],
      await rootBundle.loadString('assets/licenses/pretendard-OFL.txt'),
    );
    yield const LicenseEntryWithLineBreaks(
      ['우리강산 푸른숲체 (YK Green Forest)'],
      '유한킴벌리 "우리강산 푸른숲체(YK Green Forest)".\n'
      '유한킴벌리가 무료로 배포하는 글꼴이며, 저작권은 유한킴벌리에 있습니다.\n'
      '글꼴 자체의 유료 판매·양도는 금지됩니다.\n'
      '자세한 이용 조건은 유한킴벌리 공식 안내를 따릅니다.',
    );
    yield const LicenseEntryWithLineBreaks(
      ['AI 재질 분류 모델 — 학습 데이터'],
      '본 서비스의 재질 분류 모델은 과학기술정보통신부와 '
      '한국지능정보사회진흥원(NIA)이 지원하는 AI 통합 플랫폼 AI-Hub '
      '(aihub.or.kr)의 학습용 데이터를 활용하여 학습되었습니다.\n'
      '  · 생활폐기물 활용·환류 데이터 (AI-Hub)\n'
      '  · 재활용 품목 이미지 데이터 (AI-Hub)\n'
      '\n'
      '그 밖에 다음 공개 데이터셋을 함께 활용했습니다.\n'
      '  · TACO: Trash Annotations in Context (CC BY 4.0)\n'
      '  · Open Images Dataset (annotations CC BY 4.0)\n'
      '\n'
      '지역별 배출 정보 출처: 행정안전부 "전국생활쓰레기배출정보표준데이터" '
      '(공공데이터포털 data.go.kr).',
    );
  });
}
