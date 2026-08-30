/// 분석 실패 상태 + 다시 시도.
library;

import 'package:flutter/material.dart';

import '../../../theme/app_theme.dart';

class ResultErrorState extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const ResultErrorState({super.key, required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      color: cs.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(kSpaceL),
        child: Column(
          children: [
            Icon(Icons.error_outline, color: cs.onErrorContainer, size: 36),
            const SizedBox(height: kSpaceS),
            Text(message,
                textAlign: TextAlign.center,
                style: TextStyle(color: cs.onErrorContainer)),
            const SizedBox(height: kSpaceM),
            FilledButton.tonal(
              onPressed: onRetry,
              child: const Text('다시 시도'),
            ),
          ],
        ),
      ),
    );
  }
}
