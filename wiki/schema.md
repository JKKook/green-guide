# Wiki Schema — waste (그린가이드) 프로젝트

karpathy LLM Wiki 패턴(gist 442a6bf555914893e9891c11519de94f) 기반의 지속 위키.
이 파일의 규칙이 `/llm-wiki` 명령 본문보다 우선한다.

## 3계층 구조

1. **Raw sources (읽기 전용)** — 프로젝트 원본: 루트 `*.md` 계획 문서, `waste_app/`, `waste-api/`, `waste-classifier/`, `waste-preprocessor/`, `*_staging/`, `docs/`, `bin/`. **절대 수정 금지.**
2. **Wiki pages (`wiki/*.md`)** — 원본을 종합·압축한 지식 페이지. 자유롭게 생성/갱신/병합.
3. **Meta (`wiki/index.md`, `wiki/log.md`, `wiki/schema.md`)** — index는 전체 페이지 목록+한 줄 요약, log는 append-only 작업 이력, schema는 이 규칙.

## 페이지 규칙

- 파일명: `kebab-case.md`. 한 페이지 = 한 주제(컴포넌트, 데이터셋, 설계 결정, 프로세스 등).
- 페이지 상단에 `> 소스: <원본 경로들>` 인용 블록으로 근거 소스를 명시한다.
- 페이지 간 링크는 상대 마크다운 링크 `[페이지명](page-name.md)` 사용. 크로스링크를 적극적으로 건다.
- 언어: 한국어 본문 + 기술 용어는 원어 유지.
- 원본과 위키가 모순되면 원본이 진실. 위키를 고치고 log에 남긴다.
- 상태(진행/완료/계획)를 다룰 때는 날짜를 절대 날짜(YYYY-MM)로 적는다.

## 워크플로우

- **ingest**: 소스를 읽고 → 관련 페이지 생성/갱신(1소스가 여러 페이지를 건드릴 수 있음) → index.md 갱신 → log.md에 `## [YYYY-MM-DD] ingest | <제목>` append.
- **query**: index.md → 관련 페이지 → 필요시 원본 소스까지 파고들어 답변. 좋은 답은 새 페이지로 재적재.
- **lint**: 모순·낡은 주장·고아 페이지·누락 크로스링크 점검 후 수정, log에 append.
- log.md는 **append-only** — 기존 항목 수정/삭제 금지.
