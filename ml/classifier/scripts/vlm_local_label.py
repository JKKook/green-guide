#!/usr/bin/env python3
"""로컬 VLM 라벨러 — ollama 비전 모델로 사진에 재질 라벨(25 세부 slug)을 붙인다 (W10).

왜: 평가셋 라벨링·학습 의사라벨을 Anthropic 토큰 없이 돌릴 무료 선생님 모델.
품질은 vlm_local_eval.py 로 Opus 라벨 대비 일치율을 재서 채택 여부를 정한다.

판정 기준은 eval_v1 Opus 라벨(ai_eval_all.json)이 따른 규칙과 맞춘다:
신체·얼굴·화면 속 사진·벽/바닥 배경·노이즈 → non_object, 전자부품·배터리 내장 기기 →
외장 재질과 무관하게 electronics(보조배터리 포함), 체계 밖 물건(장난감·모형·가방) → etc,
여러 물체면 화면 면적이 가장 큰 것 하나.

입력 두 형식:
  --input <eval.jsonl>                     행마다 id·image_path (realworld_eval_set.jsonl)
  --manifest <manifest.csv> --root <dir>   file 열 = root 기준 상대경로 (oi4hn 하드 네거티브)
출력: --out JSONL (기본 outputs/logs/vlm_local/<model>/labels.jsonl). 같은 출력 파일이 캐시 —
같은 model·prompt_sha 로 이미 성공한 id 는 건너뛴다(중단 뒤 재실행 = 이어 하기).

실행 예:
  .venv/bin/python scripts/vlm_local_label.py --model qwen2.5vl:7b \
      --input ~/Downloads/greenguide-golden/realworld_eval_set.jsonl
  .venv/bin/python scripts/vlm_local_label.py --dry-run --input ...   # 모델 호출 없이 계획·프롬프트
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import _base
import requests
from greenguide_common.taxonomy import FINE_LABELS, FINE_TO_COARSE
from PIL import Image, ImageOps

DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_SIDE = 768          # 긴 변 축소 — 7B 비전 토큰 수·지연을 줄인다
RETRIES = 3
NUM_CTX = 4096          # 프롬프트+이미지 ~1k 토큰. 기본(128k)이면 KV 캐시로 7B 가 18GB 를 점유

SLUG_HINTS: dict[str, str] = {
    "paper_other": "paper: printed paper, receipts, books, paper bags, paper packaging",
    "cardboard": "corrugated cardboard boxes",
    "carton": "milk/juice cartons, sterilized drink packs (paper_pack)",
    "paper_cup": "paper cups (paper_pack)",
    "glass_brown": "brown glass bottle",
    "glass_green": "green glass bottle",
    "glass_clear": "clear glass bottle or jar",
    "glass_deposit": "Korean soju/beer deposit bottle",
    "glass_etc": "other glass (cosmetic/perfume bottles, glassware)",
    "metal": "cans, tins, aerosol cans, metal scrap, metal utensils",
    "pet": "clear PET beverage bottle",
    "plastic_other": ("other plastic containers, tubs, caps, cosmetic tubes, spray bottles, "
                      "supplement/medicine bottles (amber-tinted plastic is still plastic)"),
    "vinyl_clean": ("clean plastic film, bags, wrappers, pouches, wet-wipe packs, snack and "
                    "food packaging film (printed graphics do not make it paper)"),
    "vinyl_dirty": "food-soiled plastic film/wrappers",
    "styrofoam_white": "clean white styrofoam",
    "styrofoam_color": "colored styrofoam",
    "styrofoam_dirty": "soiled styrofoam (food trays)",
    "clothes": "clothing, textiles, blankets",
    "food_waste": ("food leftovers, fruit peels, vegetables, food sitting in or on a container, "
                   "foil or basket"),
    "electronics": ("any device with electronics or a built-in battery regardless of its "
                    "outer material: mouse, keyboard, earbuds/case, phone, laptop, hub, "
                    "smartwatch, quartz watch, power bank, small appliances, circuit boards"),
    "battery": "loose dry-cell batteries only (a power bank is electronics)",
    "trash_other": ("non-recyclable composite items for general trash: masks, rubber "
                    "slippers, stickers, tissues"),
    "light_bulb": "light bulbs, fluorescent lamps",
    "etc": ("a real object outside these categories: toys, figures, models, keychains "
            "(even device-shaped ones), bags, baskets, furniture, decor"),
    "non_object": ("no waste object: human body/face/hands, screens showing photos or "
                   "web pages, walls, floors, backgrounds, vehicles, noise images"),
}
assert set(SLUG_HINTS) == set(FINE_LABELS), "SLUG_HINTS 가 taxonomy 와 어긋남"

PROMPT = (
    "You are an expert in Korean household waste separation. Classify the single MAIN "
    "object in the photo (if several objects, the one covering the largest area).\n"
    "Answer with exactly one slug from this list:\n"
    + "\n".join(f"- {s}: {SLUG_HINTS[s]}" for s in FINE_LABELS)
    + "\n\nRules:\n"
    "1. Devices with electronics are 'electronics' even if the casing is plastic or metal.\n"
    "2. If the photo mainly shows a person, a body part, a screen image or plain background, "
    "answer 'non_object'.\n"
    "3. A real object that fits no material category is 'etc'.\n"
    "4. If food is visible, answer 'food_waste' even when it sits in a metal, foil or "
    "plastic container.\n"
    "5. Glossy printed packs (wet wipes, snacks) are plastic film -> vinyl, not paper. "
    "Judge glass only if the material is clearly glass.\n"
    "6. confidence is your probability (0.0-1.0) that the slug is correct.\n"
    'Output JSON only: {"what": "<short item description>", "slug": "<slug>", '
    '"confidence": <0.0-1.0>}'
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "what": {"type": "string"},
        "slug": {"type": "string", "enum": list(FINE_LABELS)},
        "confidence": {"type": "number"},
    },
    "required": ["what", "slug", "confidence"],
}


def load_items(input_path: Path | None, manifest: Path | None,
               root: Path | None) -> list[tuple[str, Path]]:
    """(id, 이미지 경로) 목록. jsonl 은 id·image_path, manifest 는 file 열(root 기준).

    manifest 는 근사중복 제거로 root 에 없는 행이 섞여 있어(fine-staging 기준 6,807 중 3,560)
    root 에 실제로 있는 행만 쓴다.
    """
    if input_path is not None:
        rows = [json.loads(ln) for ln in input_path.read_text().splitlines() if ln.strip()]
        return [(r["id"], Path(r["image_path"]).expanduser()) for r in rows]
    if manifest is None or root is None:
        raise SystemExit("ERROR: --input 또는 --manifest + --root 가 필요")
    with manifest.open(newline="") as f:
        rows = [(r["file"], root / r["file"]) for r in csv.DictReader(f)]
    kept = [(i, pth) for i, pth in rows if pth.exists()]
    print(f"[manifest] rows={len(rows)} present_under_root={len(kept)}")
    return kept


def encode_image(path: Path, max_side: int = MAX_SIDE) -> str:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def parse_answer(text: str) -> dict:
    """모델 응답 JSON → {fine, coarse, confidence, what}. 형식 위반은 ValueError."""
    d = json.loads(text)
    slug = d.get("slug")
    if slug not in FINE_TO_COARSE:
        raise ValueError(f"모르는 slug: {slug!r}")
    conf = float(d.get("confidence", 0.0))
    if conf > 1.0:          # 일부 모델이 0~100 으로 답한다
        conf = conf / 100.0
    return {"fine": slug, "coarse": FINE_TO_COARSE[slug],
            "confidence": round(min(max(conf, 0.0), 1.0), 3), "what": str(d.get("what", ""))}


def ask(host: str, model: str, image_b64: str, timeout: float,
        num_thread: int | None = None) -> tuple[dict, dict]:
    """ollama /api/chat 1회. (파싱 결과, 계측) 반환."""
    body = {
        "model": model, "stream": False, "format": FORMAT_SCHEMA, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 200, "num_ctx": NUM_CTX,
                    **({"num_thread": num_thread} if num_thread else {})},
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
    }
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    meta = {k: j.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                  "eval_count", "eval_duration")}
    return parse_answer(j["message"]["content"]), meta


def label_one(host: str, model: str, path: Path, timeout: float,
              retries: int = RETRIES, ask_fn=ask, num_thread: int | None = None) -> dict:
    """재시도 포함 1장 라벨. 실패하면 error 키가 있는 행."""
    t0 = time.perf_counter()
    last = ""
    for attempt in range(1, retries + 1):
        try:
            ans, meta = ask_fn(host, model, encode_image(path), timeout, num_thread)
            return {**ans, "attempts": attempt, "latency_s": round(time.perf_counter() - t0, 2),
                    "ollama": meta}
        except (requests.RequestException, ValueError, KeyError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 10))
    return {"error": last, "attempts": retries, "latency_s": round(time.perf_counter() - t0, 2)}


def load_cache(out: Path, model: str) -> set[str]:
    if not out.exists():
        return set()
    done = set()
    for ln in out.read_text().splitlines():
        r = json.loads(ln)
        if r.get("model") == model and r.get("prompt_sha") == PROMPT_SHA and "error" not in r:
            done.add(r["id"])
    return done


def ollama_ps(host: str, model: str) -> dict | None:
    """로드된 모델의 메모리 점유(size·size_vram, 바이트)."""
    try:
        for m in requests.get(f"{host}/api/ps", timeout=5).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return {"size": m.get("size"), "size_vram": m.get("size_vram")}
    except requests.RequestException:
        pass
    return None


def main() -> None:
    p = _base.make_parser("vlm_local_label", __doc__.splitlines()[0], seed=None,
                          dry_run=True, cap=True)
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--input", type=Path, help="eval jsonl (id·image_path)")
    p.add_argument("--manifest", type=Path, help="manifest.csv (file 열)")
    p.add_argument("--root", type=Path, help="manifest file 열의 기준 디렉터리")
    p.add_argument("--out", type=Path)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--num-thread", type=int, help="ollama CPU 스레드 상한 (학습과 경합 시 낮춤)")
    a = p.parse_args()

    items = load_items(a.input, a.manifest, a.root)
    out = a.out or (_base.PROJECT_ROOT / "outputs" / "logs" / "vlm_local"
                    / a.model.replace(":", "_").replace("/", "_") / "labels.jsonl")
    done = load_cache(out, a.model)
    todo = [(i, pth) for i, pth in items if i not in done]
    if a.cap:
        todo = todo[: a.cap]
    missing = [str(pth) for _, pth in todo if not pth.exists()]
    print(f"[plan] model={a.model} prompt_sha={PROMPT_SHA} items={len(items)} "
          f"cached={len(done)} todo={len(todo)} missing_files={len(missing)} out={out}")
    if missing:
        print("  missing 예:", missing[:3])
    if a.dry_run:
        print("─── prompt ───\n" + PROMPT)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    n_err = 0
    with out.open("a") as f:
        for k, (iid, pth) in enumerate(todo, 1):
            if not pth.exists():
                rec = {"error": "file missing", "attempts": 0, "latency_s": 0.0}
            else:
                rec = label_one(a.host, a.model, pth, a.timeout, num_thread=a.num_thread)
            rec = {"id": iid, "model": a.model, "prompt_sha": PROMPT_SHA, **rec}
            n_err += "error" in rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(todo)}] {iid} → {rec.get('fine', 'ERR')} "
                  f"{rec.get('confidence', '')} {rec['latency_s']}s", flush=True)
    mem = ollama_ps(a.host, a.model)
    print(f"[done] labeled={len(todo) - n_err} errors={n_err} memory={mem}")
    if mem:
        (out.parent / "memory.json").write_text(json.dumps(mem))
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
