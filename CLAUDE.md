# green-guide 공통 규칙 (모든 세션 공통)

이 저장소는 여러 Claude 세션이 **같은 작업 트리**를 동시에 사용한다.
아래 규칙은 폴더별 세션(waste_app · waste-api · waste-classifier · waste-preprocessor)과
상위 구조 세션 모두에 적용된다.

## 브랜치 전략 — `feature/* → develop → main`

| 브랜치 | 역할 | 누가 커밋하나 |
| --- | --- | --- |
| `main` | 배포본. 스토어에 올라간 버전만, 태그(`v1.0.0-beta.1` …) | **직접 커밋 금지** — 배포 시 `develop` 을 병합 |
| `develop` | 통합 브랜치. 공유 작업 트리는 항상 이 브랜치가 체크아웃돼 있다 | feature 병합만 |
| `feature/<scope>-<topic>` | 실제 작업 | 각 세션 |

- **`main`·`develop` 에 직접 커밋하지 않는다.** 작업은 반드시 `feature/*` 에서.
- 여러 세션이 한 작업 트리를 쓰므로 **브랜치를 `git checkout` 으로 바꾸지 않는다** (다른 세션의 미커밋 변경이 깨진다).
  대신 **git worktree** 를 쓴다:
  ```bash
  git worktree add .worktrees/feature-<scope>-<topic> -b feature/<scope>-<topic> develop
  # 그 경로 안에서 편집·테스트·커밋
  git -C .worktrees/feature-<scope>-<topic> push -u origin feature/<scope>-<topic>
  ```
  끝나면 `develop` 으로 PR(또는 `git merge --no-ff`) → worktree 제거(`git worktree remove`).
- scope 는 폴더명: `feature/app-…`, `feature/api-…`, `feature/classifier-…`, `feature/preprocessor-…`, `feature/repo-…`.
- 긴급 수정은 `main` 에서 `hotfix/*` 분기 → `main` 과 `develop` 양쪽 병합.

### 이미 `main` 에 직접 올라간 커밋(2026-08-30 이전)
이전 규칙 부재로 `main` 에 쌓인 리팩토링 커밋은 그대로 두고, `develop` 을 `main` 과 같은 지점으로 맞춘 뒤부터 위 규칙을 적용한다.

## 커밋 규칙
- `/commit` 스킬 사용. 메시지는 이 레포 히스토리 스타일을 따른다:
  `type(scope): 한글 제목 — 요약` + 빈 줄 + 불릿 본문 + 트레일러(`Co-Authored-By`, `Claude-Session`).
- type: `feat` `fix` `refactor` `docs` `test` `chore`. scope 는 폴더명(`waste_app`, `api`, `classifier`, `preprocessor`, `wiki`).
- **자기 scope 밖 파일은 커밋하지 않는다.** `git add -A` 금지 — 경로를 명시해서 add.
- 커밋하지 말 것: `.env*`, `key.properties`, `*.jks`, 모델 가중치(`*.onnx` 등), staging 데이터, `.venv`, `build/`.

## 세션 간 경계
| 세션 | 담당 범위 |
| --- | --- |
| waste_app | `apps/mobile/` 내부 |
| waste-api | `services/inference-api/` 내부. 모노레포 편입 — 자체 git 없음. HF 배포는 `git subtree push --prefix=services/inference-api hf main` (사용자 요청 시) |
| waste-classifier | `ml/classifier/` 내부 |
| waste-preprocessor | `ml/preprocessor/` 내부 |
| 상위 구조(repo) | 루트 파일, `docs/`, `wiki/`, `.github/`, `libs/waste-common/`(공통 패키지 — 변경은 사용 세션과 합의) |

- 다른 범위의 파일을 고쳐야 하면 직접 수정하지 말고 담당 세션(또는 사용자)에게 요청한다.
- 폴더 구조(`apps/` `services/` `ml/` `libs/`)는 2026-08-30 확정. 폴더 이동·이름 변경은 상위 구조 세션만 한다.

## 검증
- 커밋 전 해당 범위의 검증을 통과시킨다: Flutter `flutter analyze` + `flutter test`, Python `pytest`(네트워크 불필요 테스트), 서버는 `py_compile` 이상.
- 실기기 설치·HF Space 배포·Supabase 데이터 변경은 **사용자 요청이 있을 때만**.
