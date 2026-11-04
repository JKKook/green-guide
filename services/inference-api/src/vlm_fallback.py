"""Claude VLM 폴백 — 저확신 케이스만 위임하는 최후 판정자 (청사진 v2 트랙 A2).

설계 (사용자 승인 구조):
- 발동 조건: 계층 게이트 reject 또는 대분류 확신 < FALLBACK_CONF (호출부 판단)
- 모델: Claude Opus 5.5 (2026-10-31 사용자 결정 — 이 프로젝트 VLM 최소 등급 Sonnet).
  thinking 은 끌 수 없어 effort=low + max_tokens 여유로 판정 JSON 이 잘리지 않게 한다
- 이미지 768px 리사이즈 (비전 토큰 ~800), 프롬프트 캐싱으로 taxonomy 반복분 절감
- 비용 가드: 일일 호출 상한 (기본 200회, VLM_DAILY_CAP env) — 카운터 파일 영속
- 키 없으면 자동 비활성 (available=False) — 기존 경로 무영향
- 결과는 evidence(type='vlm') 로 표면화 + 자동 라벨 축적용 로그 기록
"""
from __future__ import annotations

import base64
import io
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton
from src.streams import is_valid_stream, prompt_lines
from src import disposal_v2

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COUNTER_PATH = _PROJECT_ROOT / "local_feedback" / "vlm_calls.json"
_LOG_PATH = _PROJECT_ROOT / "local_feedback" / "vlm_labels.jsonl"
# 사전 밖 생성 품목 후보 큐 — 빈도 상위가 품목 사전/CLIP 컨셉 승격 우선순위
_CANDIDATES_PATH = _PROJECT_ROOT / "local_feedback" / "item_candidates.jsonl"

MODEL = config.VLM_MODEL
DAILY_CAP = config.VLM_DAILY_CAP
MAX_SIDE = 768


class VlmFallback:
    """Anthropic Messages API 로 재질 분류 — 실패·한도초과 시 None (fail-open)."""

    def __init__(self) -> None:
        self.available = False
        # 키 상태 — /health 노출용 (키 값은 절대 담지 않는다)
        #   missing: 미설정 / invalid: 401·403 / unverified: 점검 실패(네트워크 등) / ok
        self.key_status = "missing"
        self._client = None
        key = config.ANTHROPIC_API_KEY
        if not key:
            log.info("ANTHROPIC_API_KEY 미설정 — 폴백 비활성")
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:  # noqa: BLE001
            self.key_status = "unverified"
            log.warning(f"VLM 초기화 실패 — 폴백 비활성: {exc}")
            return
        self.available = True
        self._verify_key(anthropic)
        if self.available:
            log.info(f"폴백 활성 (model={MODEL}, 일일 상한 {DAILY_CAP}, 키 {self.key_status})")

    def _verify_key(self, anthropic: Any) -> None:
        """기동 시 1회 GET /v1/models — 무효 키를 조용한 fail-open 으로 두지 않는다.

        2026-08-29 .env 재작성 뒤 키가 401 인 채로 두 달간 폴백이 한 번도 돌지 않았는데
        실패 로그가 INFO 라 아무도 몰랐다(W6). 401/403 이면 비활성 + WARNING,
        네트워크 등 판단 불가면 활성 유지(호출 시 재시도) + WARNING.
        """
        try:
            self._client.with_options(timeout=5.0, max_retries=0).models.list(limit=1)
            self.key_status = "ok"
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            self.available = False
            self.key_status = "invalid"
            log.warning(f"VLM 키 무효 — 폴백 비활성 (HTTP {exc.status_code})")
        except Exception as exc:  # noqa: BLE001
            self.key_status = "unverified"
            log.warning(f"VLM 키 점검 실패(활성 유지, 호출 시 재시도): {str(exc)[:100]}")

    # ── 비용 가드 ──────────────────────────────────────────────────────────
    @staticmethod
    def _calls_today() -> int:
        try:
            d = json.loads(_COUNTER_PATH.read_text())
            return d.get(str(date.today()), 0)
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _bump_calls() -> None:
        today = str(date.today())
        n = VlmFallback._calls_today() + 1
        _COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _COUNTER_PATH.write_text(json.dumps({today: n}))

    # ── 프롬프트 ──────────────────────────────────────────────────────────
    @staticmethod
    def _taxonomy_prompt(fine_labels: list[str], fine_to_coarse: dict[str, str]) -> str:
        pairs = [f"- {f} (대분류 {fine_to_coarse.get(f, '?')})"
                 for f in fine_labels if f not in ("non_object",)]
        return (
            "당신은 한국 분리배출 전문가입니다. 사진 속 주 물체 하나를 판정하세요.\n\n"
            "1단계 — 재질 분류가 가능하면 fine 슬러그로 답합니다.\n"
            "선택 가능한 fine 슬러그 목록:\n" + "\n".join(pairs) +
            "\n\n2단계 — 물체는 알아봤지만 위 목록에 맞는 슬러그가 없으면(소파·화분·"
            "페인트통 등) slug 를 null 로 두고, 품목명을 직접 지어 배출 스트림을 "
            "아래 목록에서만 고릅니다. 스트림을 새로 만들지 마세요.\n"
            "배출 스트림 목록:\n" + prompt_lines() +
            "\n\n규칙:\n"
            "1. 사진의 가장 주된 폐기물 하나만 판정\n"
            "2. 재질도 품목도 확실하지 않으면 slug=etc\n"
            "3. 폐기물이 아예 없으면(손·배경만) slug=non_object\n"
            "4. 품목 판정 시 상태 조건이 안내를 바꾸면 condition 에 한 줄 "
            "(예: 도자기 화분이면 불연 마대, 플라스틱이면 재활용)\n"
            + (VlmFallback._v2_prompt() if VlmFallback._mode() == "single" else
               "5. 반드시 JSON 만 출력: "
               '{"slug": "<fine 슬러그|null>", "item_name": "<품목명|null>", '
               '"stream": "<스트림 슬러그|null>", "condition": "<조건|null>", '
               '"confidence": 0.0~1.0, "reason": "<한 줄>"}')
        )

    @staticmethod
    def _mode() -> str:
        """v2 축 질의 방식 — 공통 패키지가 없거나 값이 이상하면 off(v1 그대로)."""
        m = config.VLM_TAXONOMY
        if m not in ("off", "single", "two_stage"):
            log.warning(f"GREENGUIDE_VLM_TAXONOMY={m!r} 미지원 — off 로 동작")
            return "off"
        return m if disposal_v2.AVAILABLE else "off"

    @staticmethod
    def _v2_axes_rules(n: int) -> str:
        """v2 축 판정 규칙(n, n+1, n+2 번).

        어휘는 greenguide_common.disposal 에서 가져온다(TAXONOMY_V2.md §1·§3-1).
        """
        splits = "\n".join(f"- 대분류 {c} → {' | '.join(v)}"
                           for c, v in disposal_v2.FINE_V2_SPLITS.items())
        return (
            f"{n}. 세부 분기(fine_v2) — slug 의 대분류가 아래에 있으면 둘 중 하나, 아니면 null:\n"
            + splits +
            "\n   metal_can = 음료·통조림·부탄가스·스프레이 캔, "
            "metal_scrap = 그 밖의 고철(냄비·공구·철사 등).\n"
            "   food_inedible = 동물이 먹을 수 없거나 딱딱한 것(조개·게 껍데기, 큰 뼈, 복숭아 씨, "
            "달걀·견과 껍데기, 양파·옥수수 겉껍질, 티백) — 종량제 대상.\n"
            f"{n + 1}. 상태(state) — " + " | ".join(disposal_v2.STATES) + " 중 하나: "
            "dirty=음식물·기름 오염, broken=깨짐·파손, composite=여러 재질이 붙어 분리 불가(코팅지·"
            "영수증·칫솔·고무장갑 등). 내용물이 남은 것은 상태로 보지 않습니다.\n"
            f"{n + 2}. 분리배출 표시(mark) — 제품의 분리배출 마크가 사진에서 직접 읽힐 때만 "
            + " | ".join(disposal_v2.MARKS) + " 중 하나, 안 보이면 null(추측 금지).\n"
        )

    @staticmethod
    def _v2_prompt() -> str:
        """single 모드 — 재질 판정 규칙 1~4 뒤에 v2 축 5~7 과 확장 JSON 을 붙인다."""
        return (
            VlmFallback._v2_axes_rules(5) +
            "8. 반드시 JSON 만 출력: "
            '{"slug": "<fine 슬러그|null>", "fine_v2": "<세부 분기|null>", '
            '"state": "<상태|null>", "mark": "<표시|null>", '
            '"item_name": "<품목명|null>", '
            '"stream": "<스트림 슬러그|null>", "condition": "<조건|null>", '
            '"confidence": 0.0~1.0, "reason": "<한 줄>"}'
        )

    @staticmethod
    def _axes_prompt() -> str:
        """two_stage 2회차 system — 재질은 이미 정해졌고 세부·상태·마크만 묻는다(재질 재판정 금지).

        재질 slug 는 user 메시지로 넘겨 system 을 고정(캐시 가능)으로 둔다.
        """
        return (
            "당신은 한국 분리배출 전문가입니다. 사진 속 주 물체의 재질은 이미 판정되어 "
            "사용자 메시지로 주어집니다. 재질은 바꾸지 말고 아래 세 가지만 답하세요.\n\n"
            + VlmFallback._v2_axes_rules(1) +
            (VlmFallback._toy_rule() if config.VLM_TOY else
             '4. 반드시 JSON 만 출력: {"fine_v2": "<세부 분기|null>", '
             '"state": "<상태|null>", "mark": "<표시|null>"}')
        )

    @staticmethod
    def _toy_rule() -> str:
        """GREENGUIDE_VLM_TOY=1 일 때만 2회차에 붙는 장난감 질의(TAXONOMY_V2 §3, 11-03 사용자)."""
        return (
            "4. 장난감(toy) — 놀이용으로 만든 물건"
            "(레고·블록·프라모델·플라스틱 인형·피규어)이면 true. "
            "병·컵·용기·포장재·생활용품은 캐릭터 그림이나 장난감 모양이어도 false. "
            "확실하지 않으면 false.\n"
            '5. 반드시 JSON 만 출력: {"fine_v2": "<세부 분기|null>", '
            '"state": "<상태|null>", "mark": "<표시|null>", "toy": true|false}'
        )

    @staticmethod
    def _parse_response(text: str, fine_labels: list[str],
                        fine_to_coarse: dict[str, str] | None = None) -> dict[str, Any] | None:
        """VLM 텍스트 → 검증된 판정 dict | None. 네트워크 무관(단위 테스트 대상).

        허용되는 두 형태:
        - fine 판정: slug ∈ fine_labels (single 모드면 v2 축 포함)
        - 품목 생성: slug 없음 + item_name + stream ∈ 닫힌 스트림 목록
        """
        out = VlmFallback._load_json(text)
        slug = out.get("slug")
        item_name = (out.get("item_name") or "").strip() or None
        stream = out.get("stream")
        base = {
            "confidence": float(out.get("confidence", 0.5)),
            "reason": str(out.get("reason", ""))[:120],
            "condition": (str(out.get("condition"))[:120]
                          if out.get("condition") else None),
        }
        if slug in fine_labels:
            v2 = (VlmFallback._parse_v2(out, slug, fine_to_coarse)
                  if VlmFallback._mode() == "single" else {})
            return {"slug": slug, "item_name": None, "stream": None, **base, **v2}
        if item_name and stream and is_valid_stream(str(stream)):
            return {"slug": None, "item_name": item_name[:60],
                    "stream": str(stream), **base}
        log.info(f"검증 실패 slug={slug!r} stream={stream!r} — 무시")
        return None

    @staticmethod
    def _load_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return json.loads(text)

    @staticmethod
    def _parse_v2(out: dict[str, Any], slug: str,
                  fine_to_coarse: dict[str, str] | None) -> dict[str, Any]:
        """v2 축 검증 — 목록 밖 값은 그 축만 null(재질 판정 자체는 버리지 않는다)."""
        coarse = (fine_to_coarse or {}).get(slug, slug)
        fine_v2 = out.get("fine_v2")
        if fine_v2 not in disposal_v2.FINE_V2_SPLITS.get(coarse, ()):
            fine_v2 = None
        state = out.get("state")
        if state not in disposal_v2.STATES:
            state = None
        mark = out.get("mark")
        if mark not in disposal_v2.MARKS:
            mark = None
        return {"coarse": coarse, "fine_v2": fine_v2, "state": state, "mark": mark}

    @staticmethod
    def _parse_axes(text: str | None, slug: str,
                    fine_to_coarse: dict[str, str] | None) -> dict[str, Any]:
        """two_stage 2회차 응답 → v2 축. 실패(None·깨진 JSON)면 축만 null — 재질은 1회차 그대로."""
        try:
            out = VlmFallback._load_json(text) if text else {}
        except (ValueError, AttributeError):
            log.info("VLM 2회차(v2 축) 파싱 실패 — 축 null")
            out = {}
        out = out if isinstance(out, dict) else {}
        axes = VlmFallback._parse_v2(out, slug, fine_to_coarse)
        if config.VLM_TOY:
            axes["toy"] = out.get("toy") is True   # 문자열 "true" 등은 false — 오탐 쪽으로 보수
        return axes

    def _request(self, system_text: str, b64: str, user_text: str | None = None) -> str | None:
        """Messages API 1회 → 텍스트 | None(refusal·max_tokens). 호출마다 일일 카운터 +1."""
        content: list[dict[str, Any]] = [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }]
        if user_text:
            content.append({"type": "text", "text": user_text})
        msg = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,                     # thinking 포함 상한 — 200 이면 JSON 잘림
            output_config={"effort": "low"},     # 단일 분류 — 깊은 추론 불필요
            system=[{
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},   # 프롬프트 캐싱
            }],
            messages=[{"role": "user", "content": content}],
        )
        self._bump_calls()
        if msg.stop_reason != "end_turn":        # refusal·max_tokens → 판정 없음(fail-open)
            log.info(f"VLM 판정 없음: stop_reason={msg.stop_reason}")
            return None
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    def classify(
        self,
        image_bytes: bytes,
        fine_labels: list[str],
        fine_to_coarse: dict[str, str],
    ) -> dict[str, Any] | None:
        """사진 → 판정 dict | None(비활성·한도·실패).

        fine 판정: {slug, item_name=None, stream=None, ...}
        품목 생성: {slug=None, item_name, stream, condition, ...}
        """
        if not self.available:
            return None
        if self._calls_today() >= DAILY_CAP:
            log.info(f"일일 상한({DAILY_CAP}) 도달 — 스킵")
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            t0 = time.perf_counter()
            text = self._request(self._taxonomy_prompt(fine_labels, fine_to_coarse), b64)
            if text is None:
                return None
            result = self._parse_response(text, fine_labels, fine_to_coarse)
            if result is None:
                return None
            result["vlm_calls"] = 1
            # two_stage — 재질은 v1 프롬프트 원문 그대로 1회차, 세부·상태·마크는 짧은 2회차.
            # 채택될 재질 판정(fine 임계 이상, non_object 제외)에만 부르고, 실패하면 v2 축만 null.
            slug = result["slug"]
            if (self._mode() == "two_stage" and slug is not None and slug != "non_object"
                    and result["confidence"] >= config.VLM_MIN_CONF):
                axes_text = None
                if self._calls_today() < DAILY_CAP:
                    coarse = fine_to_coarse.get(slug, slug)
                    try:
                        axes_text = self._request(
                            self._axes_prompt(), b64,
                            f"재질 판정: {slug} (대분류 {coarse})")
                        result["vlm_calls"] = 2
                    except Exception as exc:  # noqa: BLE001
                        log.warning(f"VLM 2회차 실패 — v2 축 null: {str(exc)[:100]}")
                result.update(self._parse_axes(axes_text, slug, fine_to_coarse))
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            # 자동 라벨 축적 (검토용 — 학습 반영은 별도 판단)
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), **result},
                                   ensure_ascii=False) + "\n")
            # 사전 밖 생성 품목은 후보 큐에 별도 적재 (승격 검토용)
            if result["item_name"] is not None:
                with _CANDIDATES_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"ts": time.time(), "item_name": result["item_name"],
                         "stream": result["stream"],
                         "condition": result["condition"],
                         "confidence": result["confidence"]},
                        ensure_ascii=False) + "\n")
            return result
        except Exception as exc:  # noqa: BLE001
            log.warning(f"VLM 호출 실패 (fail-open): {str(exc)[:100]}")
            return None


@lazy_singleton
def get_vlm_fallback() -> VlmFallback:
    return VlmFallback()
