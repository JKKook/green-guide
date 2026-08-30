import 'package:flutter/material.dart';

import '../core/ui/ds_card.dart';
import '../data/haptics.dart';
import '../data/legal_terms.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';

/// 약관 및 정책 목록 — 설정 > 정보 > 약관 및 정책.
class TermsListScreen extends StatelessWidget {
  const TermsListScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 0,
        title: const Text(
          '약관 및 정책',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, kSpaceS, 20, kSpaceXL),
          children: [
            DsCard(
              clipBehavior: Clip.antiAlias,
              child: Column(
                children: [
                  for (final (i, doc) in kLegalDocs.indexed) ...[
                    if (i > 0) Container(height: 1, color: t.border),
                    InkWell(
                      onTap: () {
                        Haptics.selection();
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => TermsDetailScreen(doc: doc),
                          ),
                        );
                      },
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(15, 14, 15, 14),
                        child: Row(
                          children: [
                            Icon(
                              doc.required
                                  ? Icons.description_outlined
                                  : Icons.toggle_on_outlined,
                              size: 20,
                              color: t.accentStrong,
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    doc.title,
                                    style: const TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.w700),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    '시행일 ${doc.effectiveDate}',
                                    style: TextStyle(
                                        fontSize: 11, color: t.muted2),
                                  ),
                                ],
                              ),
                            ),
                            Icon(Icons.chevron_right, size: 16, color: t.faint),
                          ],
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 12),
            Text(
              '운영자 $kLegalOperator · 문의 $kLegalContact',
              style: TextStyle(fontSize: 11.5, color: t.muted),
            ),
          ],
        ),
      ),
    );
  }
}

/// 약관 상세 — 제목·시행일·요약 + 조문.
class TermsDetailScreen extends StatelessWidget {
  final LegalDoc doc;
  const TermsDetailScreen({super.key, required this.doc});

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final bodyColor = t.dark ? t.muted2 : const Color(0xFF424244);
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 0,
        title: Text(
          doc.title,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, kSpaceS, 20, kSpaceXXL),
          children: [
            Row(
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                  decoration: BoxDecoration(
                    color: doc.required ? t.accentChipBg : t.surface,
                    border: Border.all(
                      color: doc.required
                          ? (t.dark ? kAccent700 : kAccent300)
                          : t.border,
                    ),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    doc.required ? '필수' : '선택',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: doc.required ? t.accentChipText : t.muted2,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '시행일 ${doc.effectiveDate}',
                  style: TextStyle(fontSize: 11.5, color: t.muted),
                ),
              ],
            ),
            const SizedBox(height: 14),
            DsCard(
              tinted: true,
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              child: Text(
                doc.summary,
                style: TextStyle(
                  fontSize: 13,
                  height: 1.55,
                  fontWeight: FontWeight.w600,
                  color: t.accentChipText,
                ),
              ),
            ),
            const SizedBox(height: 22),
            for (final section in doc.sections) ...[
              Text(
                section.heading,
                style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              for (final (i, item) in section.items.indexed)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: 22,
                        child: Text(
                          section.items.length > 1 ? '${i + 1}.' : '',
                          style: TextStyle(
                            fontSize: 13,
                            height: 1.6,
                            fontWeight: FontWeight.w600,
                            color: t.muted,
                          ),
                        ),
                      ),
                      Expanded(
                        child: Text(
                          item,
                          style: TextStyle(
                              fontSize: 13, height: 1.6, color: bodyColor),
                        ),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 14),
            ],
            Container(height: 1, color: t.border),
            const SizedBox(height: 12),
            Text(
              '운영자 $kLegalOperator · 문의 $kLegalContact',
              style: TextStyle(fontSize: 11.5, color: t.muted),
            ),
          ],
        ),
      ),
    );
  }
}
