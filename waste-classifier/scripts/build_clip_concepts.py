#!/usr/bin/env python3
"""CLIP 제로샷 정체 인식 자산 빌드 — SEMANTIC_FUSION_PLAN 신호② (Phase 2).

산출물 (outputs/models/clip/):
- clip_image.onnx    : CLIP ViT-B/32 이미지 인코더 (+projection) — 서빙용.
                       입력 pixel_values (B,3,224,224) → image_embeds (B,512)
- clip_concepts.npz  : 컨셉 텍스트 임베딩 (K,512, L2 정규화) + 컨셉→fine 매핑
                       텍스트 인코더는 서빙에 불필요 — 여기서 사전계산.

컨셉 추가/수정 시 이 스크립트만 재실행 → npz 교체 (재학습 불필요).
실행: .venv/bin/python scripts/build_clip_concepts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.taxonomy import FINE_LABELS  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"
OUT_DIR = config.MODELS_DIR / "clip"

# 프롬프트 템플릿 앙상블 — 단일 문구 편향 완화 (CLIP zero-shot 표준 기법)
TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "a photo of {} on a table",
]

# 컨셉 카탈로그: (영문 정체 구문, fine slug, 한국어 표시명)
# 정체→재질 추론의 근거. 한국 생활폐기물 도메인 + 실사용 오답 분석 기반.
# 한국어 표시명은 앱 증거 배지("형태 인식: 알약 통")용 — 임베딩엔 영문만 사용.
# non_object 네거티브 컨셉 포함 — 손/책상/바닥이 폐기물 클래스로 새는 것 방지.
CONCEPTS: list[tuple[str, str, str]] = [
    # ── 종이류 ──
    ("a paper receipt", "paper_other", "영수증"),
    ("a stack of paper documents", "paper_other", "종이 문서"),
    ("a paper shopping bag", "paper_other", "종이 쇼핑백"),
    ("a book", "paper_other", "책"),
    ("a cardboard delivery box", "cardboard", "택배 상자"),
    ("a sheet of corrugated cardboard", "cardboard", "골판지"),
    ("a milk carton", "carton", "우유팩"),
    ("a juice carton pack", "carton", "주스팩"),
    ("a disposable paper coffee cup", "paper_cup", "종이컵"),
    # ── 유리 ──
    ("a green soju bottle", "glass_deposit", "소주병"),
    ("a brown beer bottle", "glass_deposit", "맥주병"),
    ("a brown glass medicine bottle", "glass_brown", "갈색 유리병"),
    ("a green glass bottle", "glass_green", "녹색 유리병"),
    ("a clear glass jar", "glass_clear", "유리 용기"),
    ("a clear glass bottle", "glass_clear", "투명 유리병"),
    ("a drinking glass cup", "glass_clear", "유리컵"),
    ("a glass cosmetic jar", "glass_etc", "화장품 유리 용기"),
    # ── 금속 ──
    ("an aluminum beverage can", "metal", "음료 캔"),
    ("a tin can of food", "metal", "통조림 캔"),
    ("a metal spray can", "metal", "스프레이 캔"),
    ("a metal bottle cap", "metal", "금속 뚜껑"),
    ("a stainless steel kitchen pot", "metal", "스테인리스 냄비"),
    ("a metal fork and spoon", "metal", "금속 식기"),
    # ── 플라스틱 ──
    ("a clear plastic water bottle", "pet", "생수 페트병"),
    ("a plastic soda bottle", "pet", "음료 페트병"),
    ("a plastic supplement pill bottle", "plastic_other", "영양제 통"),
    ("a plastic medicine bottle with a label", "plastic_other", "약통"),
    ("a plastic shampoo bottle", "plastic_other", "샴푸통"),
    ("a cosmetic pump bottle", "plastic_other", "화장품 펌프 용기"),
    ("a plastic food storage container", "plastic_other", "플라스틱 밀폐용기"),
    ("a plastic detergent bottle", "plastic_other", "세제통"),
    ("a plastic delivery food container", "plastic_other", "배달 용기"),
    ("a plastic yogurt drink bottle", "plastic_other", "요구르트병"),
    # ── 비닐/필름 ──
    ("a crumpled plastic bag", "vinyl_clean", "비닐봉지"),
    ("a snack wrapper film", "vinyl_clean", "과자 봉지"),
    ("a plastic bubble wrap", "vinyl_clean", "뽁뽁이"),
    # ── 스티로폼 ──
    ("a white styrofoam shipping box", "styrofoam_white", "스티로폼 박스"),
    ("a styrofoam food tray", "styrofoam_white", "스티로폼 트레이"),
    # ── 의류 ──
    ("a t-shirt", "clothes", "티셔츠"),
    ("a pile of folded clothes", "clothes", "의류"),
    ("a pair of jeans", "clothes", "청바지"),
    ("a knitted sweater", "clothes", "니트"),
    # ── 음식물 ──
    ("food scraps and leftovers", "food_waste", "음식물"),
    ("fruit peels", "food_waste", "과일 껍질"),
    # ── 전자제품 ──
    ("a smartphone", "electronics", "스마트폰"),
    ("a computer keyboard", "electronics", "키보드"),
    ("a computer monitor", "electronics", "모니터"),
    ("a phone charger with cable", "electronics", "충전기"),
    ("wireless earbuds", "electronics", "무선 이어폰"),
    ("a TV remote control", "electronics", "리모컨"),
    ("a laptop computer", "electronics", "노트북"),
    ("a small home appliance", "electronics", "소형가전"),
    ("a computer mouse", "electronics", "마우스"),
    # ── 유해/기타 ──
    ("AA alkaline batteries", "battery", "건전지"),
    ("a lithium battery pack", "battery", "배터리팩"),
    ("a light bulb", "light_bulb", "전구"),
    ("an LED light bulb", "light_bulb", "LED 전구"),
    ("a used tissue", "trash_other", "휴지"),
    ("a toothbrush", "trash_other", "칫솔"),
    ("a ballpoint pen", "trash_other", "볼펜"),
    ("a bottle cap made of crown cork", "trash_other", "병뚜껑"),
    # ── non_object 네거티브 (폐기물 아님 → reject 유도) ──
    ("a human hand", "non_object", "손"),
    ("a human face", "non_object", "사람"),
    ("an empty wooden desk surface", "non_object", "책상 표면"),
    ("a bare floor", "non_object", "바닥"),
    ("a wristwatch worn on a wrist", "non_object", "착용 중인 시계"),
]


def main() -> None:
    from transformers import CLIPModel, CLIPProcessor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for phrase, slug, _ko in CONCEPTS:
        assert slug in FINE_LABELS, f"taxonomy 에 없는 slug: {slug} ({phrase})"

    print(f"[clip] {CLIP_NAME} 로드 (attn=eager — ONNX export 호환)")
    model = CLIPModel.from_pretrained(
        CLIP_NAME, attn_implementation="eager", use_safetensors=True)
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    # ── 컨셉 텍스트 임베딩 (템플릿 앙상블 평균 → L2 정규화) ─────────────────
    embs = []
    with torch.no_grad():
        for phrase, _, _ko in CONCEPTS:
            texts = [t.format(phrase) for t in TEMPLATES]
            tok = processor(text=texts, return_tensors="pt", padding=True)
            # transformers 5.x: get_text_features 가 출력 객체를 반환 —
            # text_model + projection 직접 경로가 버전 안정적
            out = model.text_model(
                input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            e = model.text_projection(out.pooler_output)   # (T, 512)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            embs.append((e / e.norm()).numpy())
    concept_embs = np.stack(embs).astype(np.float32)      # (K, 512)

    np.savez(
        OUT_DIR / "clip_concepts.npz",
        embeddings=concept_embs,
        phrases=np.array([p for p, _, _ in CONCEPTS]),
        phrases_ko=np.array([k for _, _, k in CONCEPTS]),
        slugs=np.array([s for _, s, _ in CONCEPTS]),
        logit_scale=np.float32(model.logit_scale.exp().item()),
    )
    print(f"[clip] 컨셉 {len(CONCEPTS)}개 임베딩 → {OUT_DIR/'clip_concepts.npz'} "
          f"(logit_scale={model.logit_scale.exp().item():.1f})")

    # ── 이미지 인코더 ONNX export ────────────────────────────────────────────
    class ImageEncoder(torch.nn.Module):
        def __init__(self, m: CLIPModel) -> None:
            super().__init__()
            self.vision = m.vision_model
            self.proj = m.visual_projection

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            out = self.vision(pixel_values=pixel_values)
            emb = self.proj(out.pooler_output)
            return emb / emb.norm(dim=-1, keepdim=True)

    enc = ImageEncoder(model).eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = OUT_DIR / "clip_image.onnx"
    torch.onnx.export(
        enc, dummy, onnx_path,
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )

    # 등가성 검증
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((2, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        ref = enc(torch.from_numpy(x)).numpy()
    (got,) = sess.run(["image_embeds"], {"pixel_values": x})
    diff = float(np.abs(ref - got).max())
    print(f"[clip] ONNX → {onnx_path} | max abs diff {diff:.2e}")
    assert diff < 1e-3, diff

    # 전처리 상수 기록 (서빙이 동일 정규화 사용하도록)
    ip = processor.image_processor
    print(f"[clip] 전처리: size={ip.size} crop={ip.crop_size} "
          f"mean={ip.image_mean} std={ip.image_std}")


if __name__ == "__main__":
    main()
