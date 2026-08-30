/// 스낵바 공통 진입점 — 화면마다 `ScaffoldMessenger.of(context).showSnackBar(SnackBar(...))`
/// 를 반복하지 않는다. 모양(floating·둥근 모서리·색)은 테마 `snackBarTheme` 이 정하고,
/// 여기서는 종류(info/error)와 표시 시간만 받는다.
library;

import 'package:flutter/material.dart';

import '../../api/api_client.dart';

enum AppSnackKind { info, error }

/// [replace] 가 true 면 떠 있는 스낵바를 먼저 내린다 — 연타 카운트처럼
/// 메시지가 빠르게 갱신되는 경우에 쓴다.
void showAppSnackBar(
  BuildContext context,
  String message, {
  AppSnackKind kind = AppSnackKind.info,
  Duration? duration,
  bool replace = false,
}) {
  final messenger = ScaffoldMessenger.of(context);
  if (replace) messenger.hideCurrentSnackBar();
  messenger.showSnackBar(
    SnackBar(
      content: Text(message),
      backgroundColor: kind == AppSnackKind.error
          ? Theme.of(context).colorScheme.error
          : null,
      duration: duration ?? const Duration(milliseconds: 4000),
    ),
  );
}

/// 예외를 사용자용 한국어 문구([friendlyError])로 바꿔 에러 스낵바로 띄운다.
void showAppErrorSnackBar(BuildContext context, Object error) =>
    showAppSnackBar(context, friendlyError(error), kind: AppSnackKind.error);
