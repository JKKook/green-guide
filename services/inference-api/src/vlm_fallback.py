"""Claude VLM 폴백 — 저확신 케이스만 위임하는 최후 판정자 (청사진 v2 트랙 A2).

설계 (사용자 승인 구조):
- 발동 조건: 계층 게이트 reject 또는 대분류 확신 < FALLBACK_CONF (호출부 판단)
- 모델: Haiku 4.5 (분류 태스크 충분 + 저비용 — 장당 ~1원)
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

from src.streams import is_valid_stream, prompt_lines
from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

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
        self._client = None
        key = config.ANTHROPIC_API_KEY
        if not key:
            log.info("ANTHROPIC_API_KEY 미설정 — 폴백 비활성")
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
            self.available = True
            log.info(f"폴백 활성 (model={MODEL}, 일일 상한 {DAILY_CAP})")
        except Exception as exc:  # noqa: BLE001
            log.info(f"초기화 실패 (비활성): {exc}")

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
            "5. 반드시 JSON 만 출력: "
            '{"slug": "<fine 슬러그|null>", "item_name": "<품목명|null>", '
            '"stream": "<스트림 슬러그|null>", "condition": "<조건|null>", '
            '"confidence": 0.0~1.0, "reason": "<한 줄>"}'
        )

    @staticmethod
    def _parse_response(text: str, fine_labels: list[str]) -> dict[str, Any] | None:
        """VLM 텍스트 → 검증된 판정 dict | None. 네트워크 무관(단위 테스트 대상).

        허용되는 두 형태:
        - fine 판정: slug ∈ fine_labels
        - 품목 생성: slug 없음 + item_name + stream ∈ 닫힌 스트림 목록
        """
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        out = json.loads(text)
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
            return {"slug": slug, "item_name": None, "stream": None, **base}
        if item_name and stream and is_valid_stream(str(stream)):
            return {"slug": None, "item_name": item_name[:60],
                    "stream": str(stream), **base}
        log.info(f"검증 실패 slug={slug!r} stream={stream!r} — 무시")
        return None

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
            msg = self._client.messages.create(
                model=MODEL,
                max_tokens=200,
                system=[{
                    "type": "text",
                    "text": self._taxonomy_prompt(fine_labels, fine_to_coarse),
                    "cache_control": {"type": "ephemeral"},   # 프롬프트 캐싱
                }],
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg",
                                   "data": b64},
                    }],
                }],
            )
            self._bump_calls()
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            result = self._parse_response(text, fine_labels)
            if result is None:
                return None
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
            log.info(f"호출 실패 (fail-open): {str(exc)[:100]}")
            return None


@lazy_singleton
def get_vlm_fallback() -> VlmFallback:
    return VlmFallback()
