/// 클래스 메타데이터 (한국형 분리수거 가이드).
///
/// 서버 `/labels` 응답에서 동적으로 로드된 것을 우선 사용.
/// 서버 응답 전(또는 오프라인)에는 계층 대분류 14종 fallback 을 표시.
/// (구 6클래스 fallback 은 2026-07-13 폐기 — 기본 클래스의 정의는 계층 대분류)
library;

import 'package:flutter/material.dart';


class WasteInfo {
  final String classKey;        // 모델 라벨 slug (paper, glass_deposit, ...)
  final String displayName;     // 한글 표시명
  final IconData icon;
  final Color color;
  final String summary;
  final List<String> howTo;
  final List<String> caution;
  final String bin;
  final bool trainedInModel;    // 현재 모델이 인식 가능한지
  // 계층 (migration 008): 1=대분류, 2=세부품목
  final int level;
  final String? parentSlug;     // 세부품목 → 부모 대분류 slug
  final bool isNegativeGuidance; // ⚠️'사실 일반쓰레기' 오분리방지 항목

  const WasteInfo({
    required this.classKey,
    required this.displayName,
    required this.icon,
    required this.color,
    required this.summary,
    required this.howTo,
    required this.caution,
    required this.bin,
    this.trainedInModel = true,
    this.level = 1,
    this.parentSlug,
    this.isNegativeGuidance = false,
  });

  factory WasteInfo.fromApi(Map<String, dynamic> json) {
    return WasteInfo(
      classKey: json['slug'] as String,
      displayName: (json['display_name'] as String?) ?? json['slug'] as String,
      summary: (json['summary'] as String?) ?? '',
      bin: (json['bin'] as String?) ?? '',
      howTo: ((json['how_to'] as List?) ?? []).map((e) => e.toString()).toList(),
      caution: ((json['caution'] as List?) ?? []).map((e) => e.toString()).toList(),
      color: _parseHexColor(json['color_hex'] as String?),
      icon: iconByName(json['icon_name'] as String?),
      trainedInModel: json['trained_in_model'] as bool? ?? false,
      level: json['level'] as int? ?? 1,
      parentSlug: json['parent_slug'] as String?,
      isNegativeGuidance: json['is_negative_guidance'] as bool? ?? false,
    );
  }
}


/// 동적 클래스 레지스트리 — 앱 전역 캐시.
/// fine slug → 대분류 slug 정적 롤업 (taxonomy 미러 — 오프라인·영역분석용).
const Map<String, String> kFineToCoarse = {
  'paper_other': 'paper', 'cardboard': 'paper',
  'carton': 'paper_pack', 'paper_cup': 'paper_pack',
  'glass_brown': 'glass', 'glass_green': 'glass', 'glass_clear': 'glass',
  'glass_deposit': 'glass', 'glass_etc': 'glass',
  'metal': 'metal',
  'pet': 'plastic', 'plastic_other': 'plastic',
  'vinyl_clean': 'vinyl', 'vinyl_dirty': 'vinyl',
  'styrofoam_white': 'styrofoam', 'styrofoam_color': 'styrofoam',
  'styrofoam_dirty': 'styrofoam',
  'clothes': 'clothes', 'food_waste': 'food_waste',
  'electronics': 'electronics', 'battery': 'hazardous',
  'trash_other': 'trash', 'light_bulb': 'trash',
};


/// 세부품목(level 2) 한글 표시명 — 서버가 slug 만 내려줄 때 사용.
const Map<String, String> kFineDisplayNames = {
  'paper_other': '기타 종이',
  'cardboard': '골판지 상자',
  'carton': '종이팩',
  'paper_cup': '종이컵',
  'glass_brown': '갈색 유리병',
  'glass_green': '녹색 유리병',
  'glass_clear': '투명 유리병',
  'glass_deposit': '보증금 반환병',
  'glass_etc': '기타 유리',
  'pet': '페트병',
  'plastic_other': '기타 플라스틱',
  'vinyl_clean': '깨끗한 비닐',
  'vinyl_dirty': '오염된 비닐',
  'styrofoam_white': '흰색 스티로폼',
  'styrofoam_color': '유색 스티로폼',
  'styrofoam_dirty': '오염된 스티로폼',
  'battery': '폐건전지',
  'trash_other': '기타 일반쓰레기',
  'light_bulb': '폐형광등·전구',
  'metal': '캔·고철',
  'clothes': '의류',
  'food_waste': '음식물',
  'electronics': '전자제품',
};

/// 세부품목 전용 아이콘 — 없으면 부모 대분류 아이콘 상속.
const Map<String, IconData> kFineIcons = {
  'pet': Icons.water_drop,
  'glass_deposit': Icons.currency_exchange,
  'glass_brown': Icons.liquor,
  'glass_green': Icons.liquor,
  'glass_clear': Icons.liquor,
  'paper_cup': Icons.coffee,
  'battery': Icons.battery_alert,
  'light_bulb': Icons.lightbulb,
  'cardboard': Icons.inventory_2,
};

class WasteClassRegistry {
  static List<WasteInfo>? _classes;
  static Map<String, WasteInfo>? _byKey;

  static List<WasteInfo> get all => _classes ?? _fallbackClasses;
  static Map<String, WasteInfo> get _map =>
      _byKey ?? {for (final c in _fallbackClasses) c.classKey: c};

  /// 빈 껍데기 세부품목 → 부모 대분류의 색·안내를 상속해 한글화.
  static WasteInfo? _synthesizeFine(
      WasteInfo c, Map<String, WasteInfo> local) {
    final parentKey = c.parentSlug ?? kFineToCoarse[c.classKey];
    final parent = local[parentKey];
    if (parent == null) return null;
    return WasteInfo(
      classKey: c.classKey,
      displayName: kFineDisplayNames[c.classKey] ?? parent.displayName,
      icon: kFineIcons[c.classKey] ?? parent.icon,
      color: parent.color,
      summary: parent.summary,
      howTo: parent.howTo,
      caution: parent.caution,
      bin: parent.bin,
      trainedInModel: c.trainedInModel,
      level: 2,
      parentSlug: parentKey,
    );
  }

  static void setFromApi(List<Map<String, dynamic>> jsonList) {
    // 병합 전략 — 서버가 장애 폴백(영문 slug 그대로·안내 없음)을 내보내는
    // 경우, 그 항목은 로컬 한국어 데이터로 대체하고 서버에 없는 slug 도
    // 로컬에서 보존한다. (Supabase 제한 중 'electronics' 영문 노출 사고 교훈)
    // 세부품목(level 2)은 로컬 폴백이 없으므로 부모에서 합성한다.
    final local = {for (final c in _fallbackClasses) c.classKey: c};
    final merged = <String, WasteInfo>{};
    for (final j in jsonList) {
      var c = WasteInfo.fromApi(j);
      final l = local[c.classKey];
      // 서버가 이름은 채웠지만 아이콘·색이 비어 있으면 로컬 것을 상속
      if (l != null) {
        final noIcon = j['icon_name'] == null;
        final noColor = j['color_hex'] == null;
        if (noIcon || noColor) {
          c = WasteInfo(
            classKey: c.classKey,
            displayName: c.displayName,
            icon: noIcon ? l.icon : c.icon,
            color: noColor ? l.color : c.color,
            summary: c.summary.isEmpty ? l.summary : c.summary,
            howTo: c.howTo.isEmpty ? l.howTo : c.howTo,
            caution: c.caution.isEmpty ? l.caution : c.caution,
            bin: c.bin.isEmpty ? l.bin : c.bin,
            trainedInModel: c.trainedInModel,
            level: c.level,
            parentSlug: c.parentSlug,
            isNegativeGuidance: c.isNegativeGuidance,
          );
        }
      }
      final degraded =
          c.displayName == c.classKey || (c.bin.isEmpty && c.howTo.isEmpty);
      final resolved = !degraded
          ? c
          : (l ?? _synthesizeFine(c, local) ?? c);
      // 동일 slug 가 대분류·세부로 중복 오면 대분류(level 1)를 우선한다.
      final existing = merged[c.classKey];
      if (existing != null && existing.level == 1 && resolved.level != 1) {
        continue;
      }
      merged[c.classKey] = resolved;
    }
    for (final e in local.entries) {
      merged.putIfAbsent(e.key, () => e.value);
    }
    _classes = merged.values.toList();
    _byKey = merged;
  }

  static WasteInfo? lookup(String key) => _map[key];

  static bool get isLoaded => _classes != null;
}


/// 편의 함수 (이전 호환성)
WasteInfo? infoFor(String classKey) => WasteClassRegistry.lookup(classKey);

/// 계층 롤업 조회 — slug 가 레지스트리에 없으면(비활성 세부품목 등)
/// parentSlug 를 따라 부모 대분류 카드로 fallback.
///
/// 예: 서버가 fine=carton 을 반환했지만 carton 이 active=false 라
/// /labels 에 없음 → paper_pack(종이팩) 카드로 안내.
WasteInfo? infoForWithRollup(String classKey, {String? parentSlug}) {
  final direct = WasteClassRegistry.lookup(classKey);
  if (direct != null) return direct;
  if (parentSlug != null) {
    final parent = WasteClassRegistry.lookup(parentSlug);
    if (parent != null) return parent;
  }
  return null;
}

/// 사용자가 직접 입력한 재질명을 서버가 아는 라벨(slug)로 해석.
///
/// 서버 `/feedback` 은 등록된 slug 만 받고 그 외에는 400 을 돌려준다.
/// 입력값을 slug·한글 표시명과 대조해 매칭되면 그 slug 를, 없으면 null 을
/// 반환한다(호출부가 "기기에만 저장" 으로 처리).
String? resolveLabelSlug(String input) {
  String norm(String v) => v
      .toLowerCase()
      .replaceAll(RegExp(r'[\s·,./_-]'), '')
      .replaceAll('류', '')
      .trim();

  final q = norm(input);
  if (q.isEmpty) return null;
  for (final c in WasteClassRegistry.all) {
    if (norm(c.classKey) == q || norm(c.displayName) == q) return c.classKey;
  }
  // 세부품목은 서버 레지스트리가 로드되기 전(오프라인 첫 실행 등)에도
  // 매핑되도록 slug·표시명 양쪽을 본다.
  for (final e in kFineDisplayNames.entries) {
    if (norm(e.key) == q || norm(e.value) == q) return e.key;
  }
  return null;
}

/// 세부품목의 부모 대분류 정보 (배지 표시용). 대분류 자신이면 null.
WasteInfo? parentInfoOf(WasteInfo info) {
  if (info.level < 2 || info.parentSlug == null) return null;
  return WasteClassRegistry.lookup(info.parentSlug!);
}

/// 이전 코드가 wasteInfoByClass 맵을 직접 참조하던 부분 호환
Map<String, WasteInfo> get wasteInfoByClass =>
    {for (final c in WasteClassRegistry.all) c.classKey: c};


Color _parseHexColor(String? hex) {
  if (hex == null) return const Color(0xFF757575);
  final h = hex.replaceFirst('#', '');
  final value = int.tryParse(h, radix: 16);
  if (value == null) return const Color(0xFF757575);
  // RGB 만 받았을 경우 (6자리) → alpha 0xFF 추가
  if (h.length == 6) return Color(0xFF000000 | value);
  return Color(value);
}


/// Material 아이콘 이름 → IconData 매핑.
/// 서버에서 받은 icon_name 문자열을 위젯이 쓸 수 있는 IconData 로 변환.
IconData iconByName(String? name) {
  switch (name) {
    case 'inbox': return Icons.inbox;
    case 'wine_bar': return Icons.wine_bar;
    case 'local_drink': return Icons.local_drink;
    case 'description': return Icons.description;
    case 'recycling': return Icons.recycling;
    case 'delete': return Icons.delete;
    case 'restaurant': return Icons.restaurant;
    case 'inventory_2': return Icons.inventory_2;
    case 'shopping_bag': return Icons.shopping_bag;
    case 'checkroom': return Icons.checkroom;
    case 'local_cafe': return Icons.local_cafe;
    case 'devices': return Icons.devices;          // 전자제품
    case 'help_outline': return Icons.help_outline;
    // 계층 세부품목 (migration 008)
    case 'battery_alert': return Icons.battery_alert;   // 폐건전지
    case 'lightbulb': return Icons.lightbulb;           // 전구
    case 'water_drop': return Icons.water_drop;         // 페트병
    case 'currency_exchange': return Icons.currency_exchange; // 보증금 반환병
    case 'liquor': return Icons.liquor;                 // 유리병 색상
    case 'warning': return Icons.warning;               // 유해폐기물
    case 'coffee': return Icons.coffee;                 // 종이컵
    default: return Icons.label_outline;
  }
}


// ────────────────────────────────────────────────────────────
// Fallback (서버 응답 받기 전 임시 사용)
// ────────────────────────────────────────────────────────────
const List<WasteInfo> _fallbackClasses = [
  // 계층 taxonomy 의 대분류(level 1) 14종 스냅샷 — migration 008 과 동일 의미.
  // 구 6클래스 fallback 은 폐기됨: 이제 "기본 클래스"의 정의는 계층 대분류다.
  // 세부(fine) 는 오프라인 시 infoForWithRollup() 이 부모 대분류로 안내한다.
  WasteInfo(classKey: 'paper', displayName: '종이류', icon: Icons.description,
      color: Color(0xFF8D6E63), summary: '신문지·책·상자 등 종이류',
      bin: '종이류 분리수거함',
      howTo: ['이물질 제거', '접거나 묶어서 배출'],
      caution: ['영수증(감열지)·코팅지는 일반쓰레기']),
  WasteInfo(classKey: 'paper_pack', displayName: '종이팩', icon: Icons.local_drink,
      color: Color(0xFF8D6E63), summary: '우유팩·두유팩·종이컵',
      bin: '종이팩 전용 수거함',
      howTo: ['헹궈서 펼쳐 말리기', '일반 종이와 섞지 않기'],
      caution: ['종이류에 넣으면 재활용 불가']),
  WasteInfo(classKey: 'glass', displayName: '유리류', icon: Icons.wine_bar,
      color: Color(0xFF26A69A), summary: '음료병·소스병 등 유리병',
      bin: '유리병 전용 수거함',
      howTo: ['내용물 비우고 헹굼', '뚜껑 분리'],
      caution: ['깨진 유리는 종량제 봉투', '도자기·내열유리는 유리 아님']),
  WasteInfo(classKey: 'metal', displayName: '캔·고철', icon: Icons.blender,
      color: Color(0xFF90A4AE), summary: '음료캔·통조림·고철',
      bin: '캔류 분리수거함',
      howTo: ['내용물 비우고 헹굼', '가능하면 압착'],
      caution: ['부탄가스는 구멍 뚫어 배출']),
  WasteInfo(classKey: 'plastic', displayName: '플라스틱', icon: Icons.recycling,
      color: Color(0xFF42A5F5), summary: '페트병·플라스틱 용기',
      bin: '플라스틱 분리수거함',
      howTo: ['내용물 비우고 헹굼', '라벨 제거·압착'],
      caution: ['복합재질·장난감은 일반쓰레기']),
  WasteInfo(classKey: 'vinyl', displayName: '비닐류', icon: Icons.shopping_bag,
      color: Color(0xFF7E57C2), summary: '봉지·포장 비닐·필름',
      bin: '비닐류 분리수거함',
      howTo: ['이물질 없이 모아서 배출'],
      caution: ['오염 비닐은 일반쓰레기']),
  WasteInfo(classKey: 'styrofoam', displayName: '스티로폼', icon: Icons.inventory_2,
      color: Color(0xFFECEFF1), summary: '완충재·포장 스티로폼',
      bin: '스티로폼 전용 수거함',
      howTo: ['테이프·스티커 제거'],
      caution: ['오염·컬러 스티로폼은 일반쓰레기 가능성']),
  WasteInfo(classKey: 'clothes', displayName: '의류', icon: Icons.checkroom,
      color: Color(0xFFEC407A), summary: '옷·신발·가방·천류',
      bin: '의류 수거함',
      howTo: ['젖지 않게 배출'], caution: []),
  WasteInfo(classKey: 'food_waste', displayName: '음식물', icon: Icons.restaurant,
      color: Color(0xFF8BC34A), summary: '남은 음식·과일 껍질',
      bin: '음식물 전용 수거',
      howTo: ['물기 제거', '이물질 없이'],
      caution: ['뼈·조개껍데기·계란껍데기는 일반쓰레기']),
  // 전자제품 안내 — 환경부 폐가전제품 무상방문수거 제도 기준 (1599-0903 / 15990903.or.kr)
  WasteInfo(classKey: 'electronics', displayName: '전자제품', icon: Icons.devices,
      color: Color(0xFF5C6BC0), summary: '소형가전·휴대폰·전선 (일반쓰레기 배출 금지)',
      bin: '소형: 주민센터 소형가전 수거함 / 대형: 무상방문수거(1599-0903)',
      howTo: [
        '소형가전(마우스·이어폰·충전기 등)은 주민센터·아파트의 소형 폐가전 수거함에 배출',
        '대형가전(냉장고·세탁기·TV 등)은 환경부 무상방문수거 신청 — 1599-0903 또는 15990903.or.kr (무료)',
        '소형가전도 5개 이상 모으면 무상방문수거 신청 가능',
        '배터리·충전지는 분리해서 폐건전지 수거함으로 (화재 위험)',
      ],
      caution: [
        '종량제 봉투에 넣으면 안 됩니다 — 전자폐기물은 전용 수거 대상',
        '휴대폰·PC 등 저장장치는 개인정보 삭제 후 배출',
      ]),
  WasteInfo(classKey: 'hazardous', displayName: '유해폐기물', icon: Icons.warning,
      color: Color(0xFFD32F2F), summary: '건전지·형광등·폐의약품',
      bin: '전용 수거함 (주민센터·아파트)',
      howTo: ['종류별 전용 수거함에'],
      caution: ['일반쓰레기 혼입 금지 — 화재·오염 위험']),
  WasteInfo(classKey: 'trash', displayName: '일반쓰레기', icon: Icons.delete,
      color: Color(0xFF757575), summary: '재활용 불가 폐기물',
      bin: '종량제 봉투',
      howTo: ['종량제 봉투에 담아 배출'], caution: []),
  WasteInfo(classKey: 'etc', displayName: '기타/분류 불가', icon: Icons.help_outline,
      color: Color(0xFF9E9E9E), summary: '분류가 어려운 물건',
      bin: '재질 확인 후 배출 (대개 일반쓰레기)',
      howTo: ['재질별로 분리 가능한 부분은 분리'], caution: []),
  WasteInfo(classKey: 'non_object', displayName: '분류 대상 아님', icon: Icons.do_not_disturb,
      color: Color(0xFFBDBDBD), summary: '물체가 인식되지 않음',
      bin: '', howTo: ['물건이 잘 보이게 다시 촬영해주세요'], caution: []),
];


/// 모델 분류 대상은 아니지만 배출 안내가 필요한 카테고리 (검색 공용).
const List<WasteInfo> kExtraGuides = [
  WasteInfo(
    classKey: 'flower_pot',
    displayName: '화분',
    icon: Icons.local_florist,
    color: Color(0xFF66BB6A),
    summary: '플라스틱·도자기 화분과 흙',
    howTo: [
      '플라스틱 화분은 흙을 비우고 플라스틱으로',
      '도자기·토분은 불연성 쓰레기 (특수마대)',
      '흙은 종량제봉투 또는 화단에',
    ],
    caution: [],
    bin: '재질별 상이',
    trainedInModel: false,
  ),
  WasteInfo(
    classKey: 'battery_guide',
    displayName: '폐건전지',
    icon: Icons.battery_alert,
    color: Color(0xFFF9A825),
    summary: '건전지·충전지·보조배터리',
    howTo: [
      '주민센터·아파트 전용 수거함에 배출',
      '일반쓰레기 배출 금지 — 화재 위험',
    ],
    caution: [],
    bin: '전용 수거함',
    trainedInModel: false,
  ),
  WasteInfo(
    classKey: 'bulky_waste',
    displayName: '대형폐기물',
    icon: Icons.local_shipping_outlined,
    color: Color(0xFF78909C),
    summary: '한 변 1m 이상의 대형 생활 폐기물',
    howTo: [
      '주민센터 방문 또는 지자체 앱·홈페이지에서 배출 신고',
      '수수료 납부 후 스티커 부착',
      '지정 장소·일시에 배출',
    ],
    caution: [],
    bin: '신고 후 스티커 배출',
    trainedInModel: false,
  ),
  WasteInfo(
    classKey: 'large_furniture',
    displayName: '대형가구',
    icon: Icons.chair_outlined,
    color: Color(0xFFA1887F),
    summary: '소파·장롱·침대 등 가구류',
    howTo: [
      '대형폐기물로 배출 신고 (수수료 스티커)',
      '상태가 좋으면 재활용센터·나눔으로도 보낼 수 있어요',
    ],
    caution: [],
    bin: '대형폐기물 신고',
    trainedInModel: false,
  ),
];
