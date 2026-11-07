#!/usr/bin/env python3
"""서빙 경로 실측(realworld_eval_golden run)에서 VLM 만 로컬 ollama 로 바꿔 재는 3단 래퍼 (W10).

왜: 유료 Opus 상시 중재 대신 무료 로컬 VLM 으로 같은 서빙 경로 정확도를 재기 위해.
서빙 프로세스(CNN·DINOv2·OCR)와 7b 를 함께 띄우면 메모리가 모자라 3단으로 나눈다:
  record  서빙 실행(ollama 안 씀) — VLM 입력(크롭 바이트·라벨 목록)을 --cache 에 저장, VLM 은 None
  vlm     서빙 없이 VlmFallback.classify 만 실행(Anthropic 호출만 ollama 로 교체) → 결과 캐시
  replay  서빙 실행(ollama 안 씀) — classify 가 캐시 결과를 돌려준다 → 서버 R1·재질 규칙으로 채점
서버 코드(services/)는 고치지 않는다. 프롬프트·파싱·중재 규칙·플래그는 서버 것 그대로다.
호출 기록·일일 카운터는 --cache 로 돌려 서버 local_feedback/ 를 건드리지 않는다.
vlm 단계는 빈 메모리가 10% 아래로 내려가면 모델을 내리고 멈춘다(exit 3, 캐시는 이어 하기).

사용 (api venv 는 x86_64 — Rosetta). FLAGS = --env GREENGUIDE_VLM_TAXONOMY=two_stage ... :
  PY=services/inference-api/.venv/bin/python; W=scripts/vlm_local_serving.py
  arch -x86_64 $PY $W --phase record --cache <dir> --eval-scripts <realworld-eval scripts> -- \\
      run --vlm --api-root <api> --model-dir <v14> --golden <gate65_A.jsonl> FLAGS --name rec --out <dir>
  arch -x86_64 $PY $W --phase vlm --cache <dir> --api-root <api> FLAGS
  arch -x86_64 $PY $W --phase replay --cache <dir> --eval-scripts ... -- run ... --name <이름> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

NUM_CTX = 4096          # vlm_local_label.py 와 같은 값 — 줄이면 이미지 토큰이 잘린다(리더 판정)
NUM_PREDICT = 1024
FREE_MIN = 10           # vlm 단계: 빈 메모리(%)가 이 값 미만이면 멈춘다(11-06 리더)


def swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return next(int(ln.split()[-1].rstrip(".")) for ln in out.splitlines()
                if ln.startswith("Swapouts"))


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[-1].split()[-1].rstrip("%"))


def guard(host: str, model: str, mem: dict, path: Path) -> None:
    """빈 메모리 최저·모델 점유 최대·스왑 증가를 기록하고, FREE_MIN 미만이면 모델을 내리고 끝낸다."""
    prev = swapouts()
    while True:
        time.sleep(2)
        f = free_pct()
        mem["free_pct_min"] = min(mem.get("free_pct_min", 100), f)
        try:
            for m in requests.get(f"{host}/api/ps", timeout=5).json().get("models", []):
                mem["model_gb_max"] = max(mem.get("model_gb_max", 0), round(m["size"] / 1e9, 2))
        except requests.RequestException:
            pass
        cur = swapouts()
        mem["swapout_total"] = mem.get("swapout_total", 0) + cur - prev
        prev = cur
        path.write_text(json.dumps(mem))
        if f < FREE_MIN:
            print(f"\nMEM_STOP free {f}% (<{FREE_MIN}%) done={mem.get('done')}", flush=True)
            requests.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0},
                          timeout=30)
            os._exit(3)


def import_vf(api_root: Path):
    """서버 vlm_fallback 만 import(앱·모델 로드 없음). 키 점검을 막고 기록 경로를 cache 로."""
    os.chdir(api_root)
    sys.path.insert(0, str(api_root))
    import src.vlm_fallback as vf  # noqa: PLC0415

    def init(self) -> None:
        self.available, self.key_status, self._client = True, "local", None

    vf.VlmFallback.__init__ = init
    return vf


def key_of(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def patch_ollama(vf, host: str, model: str, cache: Path) -> None:
    """Anthropic 호출(_request)만 ollama /api/chat 으로. 기록·카운터는 cache 아래."""
    vf._COUNTER_PATH = cache / "vlm_calls.json"
    vf._LOG_PATH = cache / "vlm_labels.jsonl"
    vf._CANDIDATES_PATH = cache / "item_candidates.jsonl"
    calls = cache / "ollama_calls.jsonl"

    def request(self, system_text: str, b64: str, user_text: str | None = None) -> str | None:
        t0 = time.perf_counter()
        r = requests.post(f"{host}/api/chat", timeout=300, json={
            "model": model, "stream": False, "format": "json", "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            "messages": [{"role": "system", "content": system_text},
                         {"role": "user", "content": user_text or "", "images": [b64]}]})
        r.raise_for_status()
        j = r.json()
        self._bump_calls()
        rec = {"ts": time.time(), "prompt_tokens": j.get("prompt_eval_count"),
               "output_tokens": j.get("eval_count"), "done_reason": j.get("done_reason"),
               "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        with calls.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if (rec["prompt_tokens"] or 0) >= NUM_CTX - NUM_PREDICT:
            print(f"\n[warn] 프롬프트 {rec['prompt_tokens']} 토큰 — num_ctx {NUM_CTX} 근접", flush=True)
        if j.get("done_reason") != "stop":      # 길이 초과 → 판정 없음(서버 fail-open 과 같게)
            return None
        return j["message"]["content"].strip()

    vf.VlmFallback._request = request


class _Rows(list):
    """run_server 가 사진을 하나씩 꺼낼 때 지금 사진 id 를 cur 에 남긴다."""

    def __init__(self, rows, cur: dict) -> None:
        super().__init__(rows)
        self.cur = cur

    def __iter__(self):
        for row in super().__iter__():
            self.cur["id"] = row.get("upload_id") or row.get("id") or row.get("file")
            yield row


def run_serving(a, rest: list[str], classify, cur: dict | None = None) -> None:
    """realworld_eval_golden run 을 그대로 돌리되 VlmFallback.classify 만 바꾼다."""
    sys.path.insert(0, str(a.eval_scripts.resolve()))
    import realworld_eval_golden as rg  # noqa: PLC0415

    orig_run = rg.run_server

    def run_server(rows, api_root, *args, **kw):   # env 설정 뒤(서버 config import 전)에 교체
        import_vf(Path(api_root)).VlmFallback.classify = classify
        return orig_run(_Rows(rows, cur if cur is not None else {}), api_root, *args, **kw)

    rg.run_server = run_server
    sys.argv = ["realworld_eval_golden.py", *rest]
    rg.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("record", "vlm", "replay"), required=True)
    ap.add_argument("--cache", type=Path, required=True, help="VLM 입력·결과 캐시 폴더")
    ap.add_argument("--eval-scripts", type=Path, help="record·replay: realworld_eval_golden.py 폴더")
    ap.add_argument("--api-root", type=Path, help="vlm: 서버 루트(services/inference-api)")
    ap.add_argument("--env", action="append", default=[], help="vlm: 서버 플래그 KEY=VAL")
    ap.add_argument("--ollama-model", default="qwen2.5vl:7b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- 뒤는 realworld_eval_golden 인자")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    cache = a.cache.resolve()
    (cache / "inputs").mkdir(parents=True, exist_ok=True)

    if a.phase == "record":
        order: list[str] = []
        cur: dict = {}
        ids: list[dict] = []    # 사진 id ↔ 캐시 키 (order 와 같은 순서)

        def record(self, image_bytes, fine_labels, fine_to_coarse):
            k = key_of(image_bytes)
            (cache / "inputs" / f"{k}.bin").write_bytes(image_bytes)
            (cache / "labels.json").write_text(json.dumps(
                {"fine_labels": list(fine_labels), "fine_to_coarse": dict(fine_to_coarse)}))
            order.append(k)
            ids.append({"upload_id": cur.get("id"), "key": k})
            return None

        run_serving(a, rest, record, cur)
        (cache / "order.json").write_text(json.dumps(order))
        (cache / "ids.json").write_text(json.dumps(ids))
        print(f"[record] VLM 입력 {len(order)}건 → {cache}")
        return

    if a.phase == "replay":
        res = cache / "results"
        hits: list[str] = []

        def replay(self, image_bytes, fine_labels, fine_to_coarse):
            f = res / f"{key_of(image_bytes)}.json"
            if not f.exists():
                raise SystemExit(f"캐시 없음: {f.name} — vlm 단계를 먼저 끝낼 것")
            hits.append(f.name)
            r = json.loads(f.read_text())
            return None if r.get("result") is None else r["result"]

        run_serving(a, rest, replay)
        print(f"[replay] 캐시 결과 {len(hits)}건 사용")
        return

    # vlm — 서빙 없이 classify 만
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    vf = import_vf(a.api_root.resolve())
    patch_ollama(vf, a.ollama_host, a.ollama_model, cache)
    lab = json.loads((cache / "labels.json").read_text())
    order = json.loads((cache / "order.json").read_text())
    (cache / "results").mkdir(exist_ok=True)
    mem: dict = {"model": a.ollama_model, "done": 0}
    threading.Thread(target=guard, daemon=True, args=(
        a.ollama_host, a.ollama_model, mem, cache / "memory.json")).start()
    fb = vf.VlmFallback()
    for i, k in enumerate(dict.fromkeys(order), 1):
        out = cache / "results" / f"{k}.json"
        if not out.exists():
            t0 = time.perf_counter()
            r = fb.classify((cache / "inputs" / f"{k}.bin").read_bytes(),
                            lab["fine_labels"], lab["fine_to_coarse"])
            out.write_text(json.dumps({"result": r, "wall_ms": round(
                (time.perf_counter() - t0) * 1000, 1)}, ensure_ascii=False))
            print(f"[{i}/{len(order)}] {k} → {r and (r.get('slug') or r.get('item_name'))}",
                  flush=True)
        mem["done"] = i
    (cache / "memory.json").write_text(json.dumps(mem))
    print("memory:", mem)


if __name__ == "__main__":
    main()
