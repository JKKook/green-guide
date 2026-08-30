import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../data/haptics.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import '../result/result_modal.dart';

/// 갤러리 선택 확인 — 시안 16d: 선택한 사진 + "재질 분석 시작".
/// (앨범 그리드는 시스템 피커가 담당하고, 선택 결과를 여기서 확인한다)
class GalleryConfirmScreen extends StatefulWidget {
  final File image;
  const GalleryConfirmScreen({super.key, required this.image});

  @override
  State<GalleryConfirmScreen> createState() => _GalleryConfirmScreenState();
}

class _GalleryConfirmScreenState extends State<GalleryConfirmScreen> {
  late File _image = widget.image;
  bool _analyzing = false;

  Future<void> _repick() async {
    Haptics.selection();
    final picked = await ImagePicker().pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
      maxWidth: 1600,
    );
    if (picked != null && mounted) setState(() => _image = File(picked.path));
  }

  Future<void> _analyze() async {
    Haptics.medium();
    setState(() => _analyzing = true);
    final close = await showResultModal(context, _image);
    if (!mounted) return;
    setState(() => _analyzing = false);
    // 분석 취소(false) — 확인 화면에 그대로 머문다. 사진을 바꾸려면
    // 상단 "다시 고르기" 로 사용자가 직접 선택 (취소했는데 시스템 피커가
    // 곧바로 다시 뜨던 문제 수정).
    if (close == false) return;
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        leading: IconButton(
          icon: const Icon(Icons.close, size: 20),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          '갤러리에서 선택',
          style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
        ),
        actions: [
          TextButton(
            onPressed: _repick,
            child: Row(
              children: [
                Text(
                  '다시 고르기',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: t.muted2,
                  ),
                ),
                Icon(Icons.expand_more, size: 13, color: t.muted2),
              ],
            ),
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: Column(
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
                child: Center(
                  child: Stack(
                    children: [
                      Container(
                        clipBehavior: Clip.antiAlias,
                        decoration: BoxDecoration(
                          border: Border.all(color: kAccent600, width: 2.5),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Image.file(_image, fit: BoxFit.contain),
                      ),
                      Positioned(
                        top: 10,
                        right: 10,
                        child: Container(
                          width: 22,
                          height: 22,
                          decoration: const BoxDecoration(
                            color: kAccent600,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.check,
                              size: 13, color: kNeutral100),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Container(
              padding: EdgeInsets.fromLTRB(
                24,
                16,
                24,
                16 + MediaQuery.viewPaddingOf(context).bottom,
              ),
              decoration: BoxDecoration(
                color: t.surface,
                border: Border(top: BorderSide(color: t.border)),
                boxShadow: [
                  BoxShadow(
                    color: kInkShadow.withValues(alpha: 0.08),
                    offset: const Offset(0, -6),
                    blurRadius: 20,
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Row(
                    children: [
                      const Text(
                        '1장 선택됨',
                        style: TextStyle(
                            fontSize: 13, fontWeight: FontWeight.w700),
                      ),
                      const Spacer(),
                      Text(
                        '업로드 → 전처리 → AI 분석 → 결과 생성',
                        style: TextStyle(fontSize: 11.5, color: t.muted),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Material(
                    color: kAccent700,
                    borderRadius: BorderRadius.circular(kRadiusMedium),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(kRadiusMedium),
                      onTap: _analyzing ? null : _analyze,
                      child: const SizedBox(
                        height: 52,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.bolt_outlined,
                                size: 17, color: kNeutral100),
                            SizedBox(width: 8),
                            Text(
                              '재질 분석 시작',
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                                color: kNeutral100,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
