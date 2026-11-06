# eval_v2 final holdout, 16 images — single final run (2026-11-06 11:18–11:21, GG leader directive with user approval)
holdout sha256 507364738ccaa8c3… (matches freeze record) · v14 · two_stage · api d3d2dc4 + uncommitted SONNET_MIN_CONF. Each configuration ran once, no reruns.

| | Current (0.55, Opus) | Sonnet always R1m + 0.6 | Opus always R1m |
|---|---|---|---|
| Coarse | 9/16 | 9/16 | 11/16 |
| Electronics FP / recall | 0/8 · 4/8 | 0/8 · 6/8 | 0/8 · 6/8 |
| reject | 5 | 4 | 4 |
| Triggered / calls | 5 / 5 | 16 / 26 | 16 / 27 |
| Cost (at each model's price) | $0.055 | $0.069 | $0.153 |
| Per 1,000 images | ~$3.4 | ~$4.3 | ~$9.6 |
| p50 | 2.0s | 5.1s | 6.7s |
| State hit / answered | 0 · 0/16 | 3 · 10/16 | 4 · 10/16 |
(The compare output's usd for Sonnet, $0.1381, is computed at Opus prices.)

## Per-image differences
- Fixed by both always modes vs current: 2a67d4ca, b14fb838 (both electronics). Both lose 0 images vs current.
- Opus only (2 images):
  - f87fa6b2 (non_object): Sonnet's response failed validation (slug and stream both null) and was ignored, so CNN's etc stood. Opus answered non_object at 0.6.
  - 49d8a1dd (trash, a Lego toy): Sonnet answered plastic_other at 0.6 and it was adopted. Opus answered trash_other at 0.8.
- Wrong in both: 0e70a79b (electronics, a blurry e-cigarette). Sonnet answered non_object at 0.6; Opus said electronics at 0.4 and was not adopted. Also 4b8c582b, b6ab46f6, 3c4ed6b5, local-58e87d0c.
- Note: the Sonnet 0.6 threshold caused the wrong answer on 49d8 (it would be discarded at 0.8, and CNN's trash would stand). The 16 holdout images are now used up.

## Sonnet 0.6 loss cases (2 images Opus got right) — for the record before the serving decision
1. 49d8a1dd6afb4819 (truth trash, orange shrimp-shaped Lego build): Sonnet answered `plastic_other` at 0.6, which passed the lowered threshold 0.6 and replaced CNN's correct trash. Opus answered `trash_other` at 0.8 (toy → trash). At the 0.8 threshold the answer is discarded and CNN's trash stands, so this loss is caused directly by lowering the threshold.
2. f87fa6b2163a4088 (truth non_object, parked car): Sonnet's 1st response failed validation with both slug and stream null and was ignored ("검증 실패 slug=None stream=None — 무시"), so CNN's etc stood. Opus answered non_object at 0.6 (non_object threshold 0.5). Not threshold-related; Sonnet's response format failed.

## Decision (2026-11-06, user via GG leader)
Serving default: Sonnet 5.5, always mode, R1, material rule, adoption threshold kept at 0.8 (0.6 not adopted, see loss case 1). Sonnet 0.8 was not measured on the holdout; the holdout is used up. api commit 38af437 (feature/api-taxonomy-v2).
