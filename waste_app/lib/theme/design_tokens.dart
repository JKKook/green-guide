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
  Color get accentChipBorder => dark ? kAccent700 : kAccent200;
  Color get accentChipText => dark ? kAccent200 : kAccent800;
  Color get bannerBg => dark ? kAccent900 : kAccent200;
}
