/// 오늘의 팁 — 홈 팁 카드와 통합 검색(가이드) 공용.
library;

const List<String> kDailyTips = [
  '페트병은 라벨을 떼고 압착한 뒤 뚜껑을 닫아 배출하면 재활용률이 올라가요',
  '기름 묻은 치킨 상자는 종이류가 아니라 일반쓰레기예요',
  '깨진 유리는 유리류가 아니에요 — 신문지에 싸서 일반쓰레기로 버려주세요',
  '우유팩은 종이류와 따로! 종이팩 전용 수거함에 배출해요',
  '스티로폼은 테이프와 송장 스티커를 떼야 재활용할 수 있어요',
  '건전지·폐의약품은 주민센터 수거함으로 — 일반쓰레기 배출 금지예요',
  '비닐봉지는 내용물을 비우고 흩날리지 않게 한데 묶어 배출해요',
];

/// 날짜 기반 오늘의 팁 — 매일 자동 교체.
String todayTip() {
  final now = DateTime.now();
  final dayOfYear = now.difference(DateTime(now.year)).inDays;
  return kDailyTips[dayOfYear % kDailyTips.length];
}

/// 오늘의 팁 배너 — 시안 13번 배너 패턴 라이브러리 (13a~13j) 렌더 에셋.
const List<String> kTipBanners = [
  'assets/banners/tip_13a.png', // 카모 블롭
  'assets/banners/tip_13b.png', // 스타버스트
  'assets/banners/tip_13c.png', // 빅 엘립스
  'assets/banners/tip_13d.png', // 웨이비 플로우
  'assets/banners/tip_13e.png', // 스퀴글
  'assets/banners/tip_13f.png', // 웨이브 위브
  'assets/banners/tip_13g.png', // 하프톤
  'assets/banners/tip_13h.png', // 아메바
  'assets/banners/tip_13i.png', // 블레이드 스트로크
  'assets/banners/tip_13j.png', // 렌즈 모자이크
];

/// 오늘의 팁 배너 — 팁이 바뀌는 날마다 날짜를 시드로 한 의사난수로 교체.
/// (하루 동안은 고정, 날이 바뀌면 달라지고 연속 이틀 같은 패턴은 피함)
String todayTipBanner() {
  final now = DateTime.now();
  final day = now.difference(DateTime(2026)).inDays;
  int pick(int d) {
    var x = d * 2654435761; // Knuth 곱셈 해시
    x ^= x >> 15;
    return x.abs() % kTipBanners.length;
  }
  var idx = pick(day);
  if (idx == pick(day - 1)) idx = (idx + 1) % kTipBanners.length;
  return kTipBanners[idx];
}
