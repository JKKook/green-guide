import 'dart:io';
import 'package:flutter/material.dart';
import '../../../api/api_client.dart';
import '../../../api/models.dart';
import '../../../core/di/app_scope.dart';
import '../../../data/haptics.dart';
import '../../../data/history_repository.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import 'feedback_sheet.dart';

/// 결과 화면 피드백 — 시안 16e "결과가 정확했나요? 정확함 / 피드백"
/// → 피드백 시 16f 시트(재질 직접 선택·입력)로 서버 전송.
/// 기록 탭 저장은 여기서만 — 사용자가 정확함/피드백으로 확정한 결과만 남긴다.
class FeedbackCard extends StatefulWidget {
  final Prediction prediction;
  final File image;

  const FeedbackCard({super.key, required this.prediction, required this.image});

  @override
  State<FeedbackCard> createState() => _FeedbackCardState();
}


class _FeedbackCardState extends State<FeedbackCard> {
  final HistoryRepository _history = AppScope.history;
  bool _sending = false;
  String? _sentLabel; // 확정된 라벨 (정확함/피드백 후 lock)
  bool _saved = false; // 기록 탭 저장 완료
  String? _error;

  /// 자유 입력이 서버 라벨로 매핑되지 않아 기기에만 저장된 경우.
  bool _localOnly = false;

  Future<void> _sendConfirm() async {
    await _send(confirmed: true, correctedLabel: null, display: null);
  }

  /// 확정 → 기록 저장(로컬, 항상) + 서버 피드백 전송(업로드 ID 있을 때만).
  Future<void> _send({
    required bool confirmed,
    required String? correctedLabel,
    required String? display,
  }) async {
    setState(() {
      _sending = true;
      _error = null;
    });
    final p = widget.prediction;
    final finalLabel = correctedLabel ?? display ?? p.predictedClass;

    // 1) 기록 — 사용자가 확정한 결과만 남긴다 (분석·탭 재분류마다 쌓이지 않게)
    try {
      await _history.save(
        sourceImage: widget.image,
        predictedClass: finalLabel,
        confidence: confirmed ? p.confidence : 1.0,
        uploadId: p.uploadId,
        modelArch: confirmed ? p.modelArch : 'user-corrected ← ${p.modelArch}',
      );
      if (mounted) setState(() => _saved = true);
    } catch (_) {
      // 기기 저장 실패(저장 공간 등) — 서버 피드백은 계속 진행
    }

    // 2) 서버 피드백 — 업로드 ID 가 없거나(온디바이스) 서버가 모르는 라벨이면
    //    로컬 기록만. (자유 입력을 그대로 보내면 서버가 400 으로 거절함)
    final unmapped = !confirmed && correctedLabel == null;
    if (p.uploadId == null || unmapped) {
      if (mounted) {
        setState(() {
          _sentLabel = finalLabel;
          _localOnly = unmapped;
          _sending = false;
        });
      }
      return;
    }
    try {
      final client = await AppScope.api();
      final result = await client.sendFeedback(
        uploadId: p.uploadId!,
        confirmed: confirmed,
        correctedLabel: correctedLabel,
      );
      if (!mounted) return;
      setState(() => _sentLabel = result.feedbackLabel);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _sentLabel = finalLabel;
        // 기록은 이미 로컬에 저장됨 — 전송만 실패했음을 짧게 알린다.
        _error = '기록은 저장했지만 서버 전송에 실패했어요. ${friendlyError(e)}';
      });
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _openFeedbackSheet() async {
    Haptics.selection();
    final choice = await showModalBottomSheet<FeedbackChoice>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => FeedbackSheet(prediction: widget.prediction),
    );
    if (choice == null) return;
    await _send(
      confirmed: false,
      correctedLabel: choice.slug,
      display: choice.display,
    );
  }

  Widget _caption(DsTokens t) {
    final text = _localOnly
        ? '기록 탭에 저장됐어요 · 목록에 없는 재질이라 서버 학습에는 반영되지 않아요'
        : _saved
            ? '기록 탭에 저장됐어요'
            : '정확함 또는 피드백을 누르면 기록 탭에 저장돼요';
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(_saved && !_localOnly ? Icons.check : Icons.info_outline,
            size: 13, color: t.muted),
        const SizedBox(width: 6),
        Flexible(
          child: Text(text,
              style: TextStyle(fontSize: 11.5, color: t.muted)),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = DsTokens.of(context);

    // 확정 완료 — 한 줄 확인 + 저장 캡션
    if (_sentLabel != null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Icon(Icons.check_circle_outline, size: 16, color: t.accentStrong),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '피드백 감사합니다 · ${infoFor(_sentLabel!)?.displayName ?? _sentLabel} 라벨로 기록',
                  style: const TextStyle(
                      fontSize: 13, fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          if (_error != null) ...[
            const SizedBox(height: 6),
            Text(_error!, style: TextStyle(color: cs.error, fontSize: 12)),
          ],
          const SizedBox(height: 14),
          _caption(t),
        ],
      );
    }

    // 시안 16e — "결과가 정확했나요?" + 정확함 / 피드백
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Expanded(
              child: Text(
                '결과가 정확했나요?',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
              ),
            ),
            Material(
              color: kAccent700,
              borderRadius: BorderRadius.circular(14),
              child: InkWell(
                borderRadius: BorderRadius.circular(14),
                onTap: _sending ? null : _sendConfirm,
                child: Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (_sending)
                        const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: kNeutral100,
                          ),
                        )
                      else
                        const Icon(Icons.thumb_up_outlined,
                            size: 14, color: kNeutral100),
                      const SizedBox(width: 6),
                      const Text(
                        '정확함',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: kNeutral100,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Material(
              color: t.surface,
              borderRadius: BorderRadius.circular(14),
              child: InkWell(
                borderRadius: BorderRadius.circular(14),
                onTap: _sending ? null : _openFeedbackSheet,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: t.handle,
                    ),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Text(
                    '피드백',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: t.body,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
        if (_error != null) ...[
          const SizedBox(height: 8),
          Text(
            _error!,
            style: TextStyle(color: cs.error, fontSize: 12),
          ),
        ],
        const SizedBox(height: 14),
        _caption(t),
      ],
    );
  }
}
