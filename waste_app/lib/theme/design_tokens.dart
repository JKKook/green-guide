import 'package:flutter/material.dart';

import 'app_theme.dart';

/// 시안(Industry 램프) 라이트 팔레트를 다크에 대응시키는 공용 토큰.
/// 시안은 라이트만 정의 — 다크는 기존 잉크 팔레트에 맞춰 한 단계씩 조정.
class DsTokens {
  final bool dark;
  const DsTokens(this.dark);

  factory DsTokens.of(BuildContext context) =>
      DsTokens(Theme.of(context).brightness == Brightness.dark);

  Color get surface => dark ? const Color(0xFF1A2027) : kNeutral100;
  Color get border => dark ? const Color(0xFF2A2F36) : kNeutral200;
  Color get muted => dark ? const Color(0xFF8C9199) : kNeutral500;
  Color get muted2 => dark ? const Color(0xFFA5ABB3) : kNeutral600;
  Color get faint => dark ? const Color(0xFF5D5D60) : kNeutral400;
  Color get accentStrong => dark ? kAccent300 : kAccent700;
  Color get accentDeep => dark ? kAccent200 : kAccent900;
  Color get accentChipBg =>
      dark ? kAccent400.withValues(alpha: 0.16) : kAccent100;
  /// 강조 패널·칩 테두리 — 화면 다수가 쓰던 kAccent300 으로 통일 (구 kAccent200).
  Color get accentChipBorder => dark ? kAccent700 : kAccent300;
  Color get accentChipText => dark ? kAccent200 : kAccent800;
  Color get bannerBg => dark ? kAccent900 : kAccent200;

  /// 시트 드래그 핸들·연한 구분선·비활성 도트.
  Color get handle => dark ? const Color(0xFF5D5D60) : kNeutral300;
  /// 본문 설명 텍스트 (라이트에서 muted 보다 한 단계 진함).
  Color get body => dark ? muted2 : const Color(0xFF5D5D60);
  /// 보조 아이콘 (라이트 kNeutral400 — muted 보다 옅음).
  Color get iconMuted => dark ? const Color(0xFF8C9199) : kNeutral400;
  /// accent-2 램프의 텍스트/아이콘.
  Color get accent2Text => dark ? kAccent2300 : kAccent2900;
  /// 은은한 accent 채움(도트·배지 배경).
  Color get accentSoft => dark ? kAccent700 : kAccent400;
}
