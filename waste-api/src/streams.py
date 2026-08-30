"""배출 스트림(목적지) 닫힌 목록 — 환경부 지침 대조 확정본 (2026-07-29).

품목은 무한하지만 배출 목적지는 유한하다는 원칙의 코드화. VLM 폴백이
사전 밖 품목을 자유 생성할 때, 배출 안내만은 이 목록에서 고르게 강제한다.

근거: 「재활용가능자원의 분리수거 등에 관한 지침」 별표 1 요약(법령정보센터),
환경부 재활용품 분리배출 가이드라인(2018), 생활계 유해폐기물 종류 고시(2022).
- 지역 편차 품목(아이스팩·LED 수거 등)은 스트림이 아니라 region_waste_rules 로 처리.
- 스트림 추가는 사람 승인으로만 (VLM 이 새 스트림을 발명할 수 없음).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stream:
    slug: str
    display_name: str
    summary: str            # VLM 답안지 겸 사용자 안내 한 줄
    how_to: list[str] = field(default_factory=list)


# slug 순서 = 프롬프트 노출 순서 (일반 → 재활용 → 전용 → 신고·방문 → 역회수)
STREAMS: dict[str, Stream] = {s.slug: s for s in [
    Stream("general_trash", "종량제 봉투",
           "재활용 불가 가연성 폐기물은 종량제 봉투로 배출",
           ["지자체 종량제 봉투 사용", "배출 요일·장소 준수"]),
    Stream("incombustible_bag", "불연성 특수마대",
           "도자기·깨진유리·화분·소량 건축폐기물 등 타지 않는 폐기물은 특수규격마대로 배출",
           ["종량제 봉투 판매소에서 특수마대 구입", "깨진 유리는 신문지에 싸서"]),
    Stream("food_waste", "음식물쓰레기 수거함/봉투",
           "음식물은 물기 제거 후 음식물 전용 수거함·봉투로 배출",
           ["물기 최대한 제거", "비닐·이쑤시개 혼입 금지"]),
    Stream("recycle_paper", "종이류 수거함",
           "신문지·책·골판지 등 종이류 (테이프·이물질 제거)",
           ["테이프·철핀 제거", "펼쳐서 묶어 배출"]),
    Stream("recycle_paper_pack", "종이팩 전용 수거함",
           "우유팩·멸균팩은 일반 종이와 별도 배출 (없으면 주민센터 교환)",
           ["헹궈서 펼쳐 말리기"]),
    Stream("recycle_glass", "유리병 수거함",
           "유리병류 (보증금 대상 제외)",
           ["내용물 비우고 헹구기", "뚜껑 분리"]),
    Stream("deposit_return", "소매점 보증금 반환",
           "소주·맥주병 등 빈용기 보증금 대상은 소매점 반환 (병당 70~130원)",
           ["뚜껑 닫아 소매점 반환"]),
    Stream("recycle_metal", "캔·고철 수거함",
           "음료캔·통조림·고철류 (부탄가스·스프레이는 내용물 완전 제거 후)",
           ["내용물 비우고 헹구기", "부탄·스프레이는 구멍 뚫어 배출"]),
    Stream("recycle_plastic", "플라스틱 수거함",
           "플라스틱 용기류 (이물질 제거)",
           ["내용물 비우고 헹구기", "라벨·뚜껑 분리"]),
    Stream("recycle_pet_clear", "투명페트 별도 배출",
           "무색 투명 PET 음료병은 별도 배출",
           ["라벨 완전 제거", "압착 후 뚜껑 닫아 배출"]),
    Stream("recycle_vinyl", "비닐 수거함",
           "깨끗한 비닐·필름류 (오염 시 종량제)",
           ["이물질 없이 모아서 배출"]),
    Stream("recycle_styrofoam", "스티로폼 전용 수거함",
           "깨끗한 흰색 스티로폼 (오염·색상은 지역 규정 확인)",
           ["테이프·스티커 제거"]),
    Stream("collection_battery", "폐건전지 수거함",
           "건전지·보조배터리 등 전지류 전용 수거함 (주민센터·아파트)",
           ["제품에서 전지 분리", "단자 절연 권장"]),
    Stream("collection_fluorescent", "폐형광등 수거함",
           "형광등 전용 수거함 (깨진 형광등은 신문지에 싸서 종량제)",
           ["깨지지 않게 배출"]),
    Stream("collection_medicine", "폐의약품 수거함",
           "폐의약품은 약국·보건소·주민센터·우체통 전용수거함으로",
           ["포장 제거 후 내용물만 모아서"]),
    Stream("collection_clothes", "의류수거함",
           "옷·이불 등 원단류 (오염·훼손 심하면 종량제)",
           ["세탁 후 배출 권장"]),
    Stream("collection_cooking_oil", "폐식용유 수거함",
           "폐식용유 전용 수거함 (아파트·주민센터, 지역별 상이)",
           ["이물질 거른 후 배출"]),
    Stream("collection_hazardous_chemical", "생활계 유해폐기물 수거함",
           "폐페인트·락카·광택제·접착제 등 인화성 폐기물은 지정 수거함으로",
           ["내용물 유출 방지 포장", "인화성 표시 후 배출"]),
    Stream("collection_small_appliance", "소형가전 수거함",
           "소형 전자제품(드라이어·가습기 등) 전용 수거함",
           ["전지·코드 분리 가능하면 분리"]),
    Stream("bulky_waste", "대형폐기물 신고 배출",
           "가구·매트리스 등 대형폐기물은 지자체 신고 후 수수료 스티커 부착 배출",
           ["지자체 앱/주민센터 신고", "수수료 납부 후 지정 장소 배출"]),
    Stream("appliance_pickup", "폐가전 무상방문수거",
           "냉장고·TV 등 대형가전은 무상방문수거(1599-0903) 또는 판매점 역회수",
           ["1599-0903 또는 온라인 예약"]),
    Stream("retailer_takeback", "판매처·정비소 역회수",
           "타이어·윤활유(정비업소), 전지(대리점) 등은 판매·정비처 반납",
           []),
    # 예약 (도시 앱 스코프 밖 — 프롬프트 비노출)
    Stream("farm_waste", "영농폐기물 공동집하장",
           "폐농약·농약용기는 마을공동집하장으로", []),
]}

# VLM 답안지에서 제외할 스트림 (스코프 밖 예약분)
_PROMPT_EXCLUDED = {"farm_waste"}


def is_valid_stream(slug: str) -> bool:
    return slug in STREAMS


def prompt_lines() -> str:
    """VLM 프롬프트용 '- slug: 설명' 목록."""
    return "\n".join(
        f"- {s.slug}: {s.display_name} — {s.summary}"
        for s in STREAMS.values() if s.slug not in _PROMPT_EXCLUDED)


def to_api_dict(slug: str) -> dict[str, object] | None:
    s = STREAMS.get(slug)
    if s is None:
        return None
    return {"slug": s.slug, "display_name": s.display_name,
            "summary": s.summary, "how_to": s.how_to}
