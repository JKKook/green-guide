/// 온보딩 ① 브랜드 소개 + 이용 동의 시트.
library;

import 'package:flutter/material.dart';

import '../../../data/haptics.dart';
import '../../../data/legal_terms.dart';
import '../../../screens/terms_screen.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import '../widgets/onboarding_primitives.dart';

class ConsentStep extends StatefulWidget {
  final Future<void> Function({required bool alarmOptIn, required bool aiOptIn})
      onDone;
  const ConsentStep({super.key, required this.onDone});

  @override
  State<ConsentStep> createState() => _ConsentStepState();
}


class _ConsentStepState extends State<ConsentStep> {
  /// 동의 항목 = 약관 문서 목록 (필수 3 · 선택 2) — 순서 고정.
  static final List<LegalDoc> _items = kLegalDocs;

  final List<bool> _checked = [false, false, false, false, false];
  bool _busy = false;

  bool get _allChecked => _checked.every((c) => c);
  bool get _requiredOk =>
      [for (final (i, it) in _items.indexed) if (it.required) _checked[i]]
          .every((c) => c);

  void _toggleAll() {
    Haptics.selection();
    final next = !_allChecked;
    setState(() {
      for (var i = 0; i < _checked.length; i++) {
        _checked[i] = next;
      }
    });
  }

  void _showDetail(int i) {
    Haptics.selection();
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => TermsDetailScreen(doc: _items[i])),
    );
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Stack(
      fit: StackFit.expand,
      children: [
        // 뒤 배경 — 브랜드 블록 (시안: blur + 55% opacity)
        Opacity(
          opacity: 0.55,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 88,
                height: 88,
                decoration: BoxDecoration(
                  color: kAccent700,
                  borderRadius: BorderRadius.circular(kRadiusXL),
                ),
                child: const Icon(Icons.recycling, size: 46, color: kNeutral100),
              ),
              const SizedBox(height: 16),
              const Text(
                '그린가이드',
                style: TextStyle(
                  fontSize: 34,
                  fontWeight: FontWeight.w600,
                  letterSpacing: -0.34,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                '사진 한 장으로 끝내는 분리배출',
                style: TextStyle(fontSize: 13, color: t.muted2),
              ),
              const SizedBox(height: 200),
            ],
          ),
        ),
        // 스크림 + 동의 시트
        Container(color: kInkShadow.withValues(alpha: 0.42)),
        Align(
          alignment: Alignment.bottomCenter,
          child: SingleChildScrollView(
            child: SheetCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '그린가이드 이용 동의',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '서비스 시작을 위해 약관에 동의해주세요',
                    style: TextStyle(fontSize: 12.5, color: t.muted2),
                  ),
                  const SizedBox(height: 18),
                  // 전체 동의
                  InkWell(
                    borderRadius: BorderRadius.circular(kRadiusMedium),
                    onTap: _toggleAll,
                    child: Container(
                      padding: const EdgeInsets.fromLTRB(16, 15, 16, 15),
                      decoration: BoxDecoration(
                        color: t.accentChipBg,
                        border: Border.all(
                          color: _allChecked
                              ? (t.dark ? kAccent500 : kAccent400)
                              : t.border,
                          width: 1.5,
                        ),
                        borderRadius: BorderRadius.circular(kRadiusMedium),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              color: _allChecked ? kAccent700 : t.surface,
                              border: _allChecked
                                  ? null
                                  : Border.all(color: t.faint, width: 1.5),
                              shape: BoxShape.circle,
                            ),
                            child: _allChecked
                                ? const Icon(Icons.check,
                                    size: 14, color: kNeutral100)
                                : null,
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            '전체 동의',
                            style: TextStyle(
                                fontSize: 14.5, fontWeight: FontWeight.w700),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  for (final (i, item) in _items.indexed)
                    InkWell(
                      borderRadius: BorderRadius.circular(kRadiusSmall),
                      onTap: () {
                        Haptics.selection();
                        setState(() => _checked[i] = !_checked[i]);
                      },
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(kSpaceL, kSpaceM, kSpaceS, kSpaceM),
                        child: Row(
                          children: [
                            Icon(
                              Icons.check,
                              size: 17,
                              color: _checked[i]
                                  ? t.accentStrong
                                  : (t.dark
                                      ? const Color(0xFF5D5D60)
                                      : kNeutral300),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text.rich(
                                TextSpan(children: [
                                  TextSpan(
                                    text: item.required ? '[필수] ' : '[선택] ',
                                    style: item.required
                                        ? TextStyle(
                                            fontWeight: FontWeight.w700,
                                            color: t.accentChipText)
                                        : null,
                                  ),
                                  TextSpan(
                                      text: item.title
                                          .replaceAll(' (선택)', '')),
                                ]),
                                style: TextStyle(
                                  fontSize: 13,
                                  color: t.dark
                                      ? t.muted2
                                      : const Color(0xFF5D5D60),
                                ),
                              ),
                            ),
                            InkWell(
                              borderRadius: BorderRadius.circular(999),
                              onTap: () => _showDetail(i),
                              child: Padding(
                                padding: const EdgeInsets.all(6),
                                child: Icon(Icons.chevron_right,
                                    size: 15, color: t.faint),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(height: 14),
                  OnboardingButton(
                    label: '동의하고 시작하기',
                    onTap: _requiredOk && !_busy
                        ? () async {
                            setState(() => _busy = true);
                            await widget.onDone(
                              alarmOptIn: _checked[3],
                              aiOptIn: _checked[4],
                            );
                          }
                        : null,
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ─── ② 지역 선택 (17b) + 시·군·구 시트 (17f) ─────────────────────────────────
