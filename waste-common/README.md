# waste-common

`waste-preprocessor` · `waste-classifier` · `waste-api` 가 공유하는 공통단. 각 프로젝트 venv 에 editable 로 설치한다.

```bash
# 각 프로젝트 requirements.txt 에 포함됨
pip install -e ../waste-common
```

| 모듈 | 역할 |
|---|---|
| `settings` | `.env` 로드(1회), 형제 프로젝트 루트(`PREPROCESSOR_ROOT` 등, env 로 override), Supabase 테이블·버킷 이름 |
| `taxonomy` | 계층 분류체계 SSOT (coarse 14 × fine 25) + `LEGACY_LABELS` — `waste-classifier/src/taxonomy.py` 에서 이동 |
| `imaging` | `decode_rgb`(EXIF 보정) → `resize_square` → `to_normalized_array`, `preprocess(src, layout=)`, `recompress_for_storage`, `content_type`, ImageNet 상수 |
| `supabase` | `get_client()`(memoized) / `try_get_client()` / `Bucket` / `upload_and_get_url` / `download` |
| `logging` | `get_logger(name)`, `fail_open(log, what)` |
| `cli` | `make_parser(prog, *, seed, dry_run, cap)` |

의존: numpy, Pillow, python-dotenv (+ optional `supabase`). **torch 는 두지 않는다** — api 런타임을 가볍게 유지하기 위함.

```bash
arch -arm64 ../waste-preprocessor/.venv/bin/python -m pytest -q   # 테스트
```
