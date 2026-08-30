/// 피드백 입력 시트 — 맞음/틀림 + 직접 입력.
library;

import 'package:flutter/material.dart';
import '../../../api/models.dart';
import '../../../data/haptics.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';

/// 피드백 시트 결과 — 서버가 아는 라벨(slug)인지, 자유 입력인지 구분.
///
/// 서버 `/feedback` 은 등록된 slug 만 받으므로(그 외 400), 매핑되지 않은
/// 자유 입력은 서버로 보내지 않고 기기 기록에만 남긴다.
class FeedbackChoice {
  /// 서버로 보낼 라벨 — 매핑 실패 시 null.
  final String? slug;

  /// 사용자가 실제로 고르거나 입력한 문구 (기록 표시용).
  final String display;

  const FeedbackChoice({required this.slug, required this.display});
}

/// 피드백 시트 — 시안 16f: 실제 재질 선택(칩) 또는 직접 입력 → 서버 전송.
class FeedbackSheet extends StatefulWidget {
  final Prediction prediction;
  const FeedbackSheet({super.key, required this.prediction});

  @override
  State<FeedbackSheet> createState() => _FeedbackSheetState();
}


class _FeedbackSheetState extends State<FeedbackSheet> {
  /// 시안 칩 순서 — (라벨, 클래스 키)
  static const List<(String, String)> _chips = [
    ('플라스틱', 'plastic'),
    ('페트병', 'pet'),
    ('비닐', 'vinyl'),
    ('캔·고철', 'metal'),
    ('유리병', 'glass'),
    ('종이', 'paper'),
    ('종이팩', 'paper_pack'),
    ('일반쓰레기', 'trash'),
  ];

  final TextEditingController _custom = TextEditingController();
  String? _selectedKey;

  @override
  void dispose() {
    _custom.dispose();
    super.dispose();
  }

  FeedbackChoice? get _result {
    final typed = _custom.text.trim();
    if (typed.isNotEmpty) {
      // 서버는 등록된 라벨만 받으므로 입력값을 slug 로 해석해 본다.
      return FeedbackChoice(slug: resolveLabelSlug(typed), display: typed);
    }
    final key = _selectedKey;
    if (key == null) return null;
    return FeedbackChoice(
        slug: key, display: infoFor(key)?.displayName ?? key);
  }

  /// 입력한 자유 텍스트가 서버가 모르는 재질인지 (안내 문구 분기용).
  bool get _typedUnmapped {
    final typed = _custom.text.trim();
    return typed.isNotEmpty && resolveLabelSlug(typed) == null;
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final predictedName = infoFor(widget.prediction.predictedClass)?.displayName ??
        widget.prediction.predictedClass;
    final canSend = _result != null;

    return SafeArea(
      child: Padding(
        padding: EdgeInsets.fromLTRB(
          kSpaceXL,
          0,
          kSpaceXL,
          kSpaceXL + MediaQuery.viewInsetsOf(context).bottom,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    '실제 재질은 무엇이었나요?',
                    style:
                        TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                ),
                Semantics(
                  button: true,
                  label: '닫기',
                  child: InkWell(
                    borderRadius: BorderRadius.circular(999),
                    onTap: () => Navigator.of(context).pop(),
                    child: Padding(
                      padding: const EdgeInsets.all(kSpaceXS),
                      child: Icon(Icons.close, size: 20, color: t.faint),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text.rich(
              TextSpan(
                children: [
                  const TextSpan(text: 'AI가 '),
                  TextSpan(
                    text: predictedName,
                    style: const TextStyle(fontWeight: FontWeight.w700),
                  ),
                  const TextSpan(text: '으로 분석했어요 · 올바른 재질을 알려주세요'),
                ],
              ),
              style: TextStyle(fontSize: 12.5, color: t.muted2),
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final (label, key) in _chips)
                  Builder(builder: (context) {
                    final selected = _selectedKey == key;
                    return InkWell(
                      borderRadius: BorderRadius.circular(999),
                      onTap: () {
                        Haptics.selection();
                        setState(() {
                          _selectedKey = selected ? null : key;
                          if (!selected) _custom.clear();
                        });
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 11),
                        decoration: BoxDecoration(
                          color: selected ? kAccent700 : t.surface,
                          border: selected
                              ? null
                              : Border.all(
                                  color: t.dark
                                      ? const Color(0xFF5D5D60)
                                      : kNeutral300),
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (selected) ...[
                              const Icon(Icons.check,
                                  size: 14, color: kNeutral100),
                              const SizedBox(width: 6),
                            ],
                            Text(
                              label,
                              style: TextStyle(
                                fontSize: 13.5,
                                fontWeight: selected
                                    ? FontWeight.w700
                                    : FontWeight.w600,
                                color: selected
                                    ? kNeutral100
                                    : Theme.of(context).colorScheme.onSurface,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              height: 54,
              padding: const EdgeInsets.only(left: kSpaceL, right: kSpaceM),
              decoration: BoxDecoration(
                color: t.surface,
                border: Border.all(
                  color: t.handle,
                ),
                borderRadius: BorderRadius.circular(kRadiusMedium),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _custom,
                      onChanged: (v) => setState(() {
                        if (v.trim().isNotEmpty) _selectedKey = null;
                      }),
                      style: const TextStyle(fontSize: 14),
                      decoration: InputDecoration(
                        isDense: true,
                        border: InputBorder.none,
                        enabledBorder: InputBorder.none,
                        focusedBorder: InputBorder.none,
                        contentPadding: EdgeInsets.zero,
                        filled: false,
                        hintText: '목록에 없으면 직접 입력 (예: 아이스팩)',
                        hintStyle: TextStyle(fontSize: 14, color: t.muted),
                      ),
                    ),
                  ),
                  Icon(Icons.edit_outlined, size: 18, color: t.faint),
                ],
              ),
            ),
            const SizedBox(height: 14),
            Material(
              color: canSend ? kAccent700 : t.border,
              borderRadius: BorderRadius.circular(kRadiusMedium),
              child: InkWell(
                borderRadius: BorderRadius.circular(kRadiusMedium),
                onTap: canSend
                    ? () {
                        Haptics.medium();
                        Navigator.of(context).pop(_result);
                      }
                    : null,
                child: SizedBox(
                  height: 56,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.send_outlined,
                          size: 17, color: canSend ? kNeutral100 : t.muted),
                      const SizedBox(width: 8),
                      Text(
                        '피드백 보내기',
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: canSend ? kNeutral100 : t.muted,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(_typedUnmapped ? Icons.smartphone : Icons.history,
                    size: 13, color: t.muted),
                const SizedBox(width: 6),
                Flexible(
                  child: Text(
                    _typedUnmapped
                        ? '목록에 없는 재질이라 기기 기록에만 저장돼요'
                        : '서버로 전송되어 AI 재질 분석 학습에 사용돼요',
                    style: TextStyle(fontSize: 11.5, color: t.muted),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
