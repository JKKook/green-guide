# scripts/archive — 완료된 1회용 데이터 통합 스크립트

데이터 통합이 끝나 더 이상 실행하지 않는 스크립트. 재현성을 위해 삭제하지 않고 보관한다.
`scripts/_base.py` 를 import 하는 파일이 있으므로 다시 실행하려면 `scripts/` 를 경로에 넣는다:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/archive/<name>.py --help
```

| 파일 | 용도 | 상태 |
|---|---|---|
| `integrate_aihub.py` | AI-Hub 71385/71647 스테이징 → manifest | 통합 완료 |
| `integrate_kaggle_garbage12.py` | Kaggle garbage-12 → manifest | 통합 완료 |
| `integrate_taco.py` | TACO → manifest | 통합 완료 |
| `extend_manifest_synthetic.py` / `extend_manifest_taco.py` | 합성·TACO 항목을 manifest 에 추가 | 통합 완료 |
| `extract_bg_140.py` | 실내 배경 140장 추출 | 완료 |
| `_tau_check.py` | OOD 임계값(τ) 1회 점검 | 완료 |
