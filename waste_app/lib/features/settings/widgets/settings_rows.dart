/// 설정 화면 구성 요소 — 섹션 라벨·설정 행.
library;

import 'package:flutter/material.dart';

import '../../../theme/design_tokens.dart';

/// 섹션 라벨 — 시안: 11px w700 accent-800.
class SettingsSectionLabel extends StatelessWidget {
  final DsTokens tokens;
  final String label;
  const SettingsSectionLabel({super.key, required this.tokens, required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 20, bottom: 8),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.22,
          color: tokens.accentChipText,
        ),
      ),
    );
  }
}


/// 설정 행 — 아이콘(accent-700) + 제목/부제 + 트레일링.
class SettingsRow extends StatelessWidget {
  final DsTokens tokens;
  final IconData icon;
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final VoidCallback? onTap;
  const SettingsRow({super.key, 
    required this.tokens,
    required this.icon,
    required this.title,
    this.subtitle,
    this.trailing,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final row = Padding(
      padding: const EdgeInsets.fromLTRB(15, 14, 15, 14),
      child: Row(
        children: [
          Icon(icon, size: 20, color: tokens.accentStrong),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                      fontSize: 14, fontWeight: FontWeight.w700),
                ),
                if (subtitle != null) ...[
                  const SizedBox(height: 2),
                  Text(
                    subtitle!,
                    style: TextStyle(fontSize: 11, color: tokens.muted2),
                  ),
                ],
              ],
            ),
          ),
          ?trailing,
        ],
      ),
    );
    if (onTap == null) return row;
    return InkWell(onTap: onTap, child: row);
  }
}
