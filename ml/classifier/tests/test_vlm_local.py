"""로컬 VLM 라벨러·평가 — 네트워크 없이 파싱·재시도·캐시·지표만 검증."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import vlm_local_eval as ev  # noqa: E402
import vlm_local_label as lb  # noqa: E402


def test_parse_answer_maps_coarse_and_clamps_percent():
    r = lb.parse_answer('{"what": "mouse", "slug": "electronics", "confidence": 92}')
    assert r == {"fine": "electronics", "coarse": "electronics", "confidence": 0.92,
                 "what": "mouse"}
    assert lb.parse_answer('{"slug": "pet", "confidence": 0.5}')["coarse"] == "plastic"


def test_parse_answer_rejects_unknown_slug():
    with pytest.raises(ValueError):
        lb.parse_answer('{"slug": "wood", "confidence": 0.9}')


def test_label_one_retries_then_succeeds(tmp_path, monkeypatch):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (1200, 800)).save(img)
    monkeypatch.setattr(lb.time, "sleep", lambda s: None)
    calls = []

    def fake(host, model, b64, timeout, num_thread=None):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("bad json")
        return lb.parse_answer('{"slug": "metal", "confidence": 0.8, "what": "can"}'), {}

    r = lb.label_one("h", "m", img, 1.0, ask_fn=fake)
    assert r["fine"] == "metal" and r["attempts"] == 2


def test_cache_skips_only_same_model_prompt_success(tmp_path):
    out = tmp_path / "l.jsonl"
    rows = [{"id": "a", "model": "m", "prompt_sha": lb.PROMPT_SHA, "fine": "metal"},
            {"id": "b", "model": "m", "prompt_sha": lb.PROMPT_SHA, "error": "x"},
            {"id": "c", "model": "other", "prompt_sha": lb.PROMPT_SHA, "fine": "metal"},
            {"id": "d", "model": "m", "prompt_sha": "old", "fine": "metal"}]
    out.write_text("\n".join(json.dumps(r) for r in rows))
    assert lb.load_cache(out, "m") == {"a"}


def test_prompt_lists_every_fine_slug():
    for s in lb.FINE_LABELS:
        assert f"- {s}:" in lb.PROMPT


def _p(coarse, conf, fine=None):
    return {"coarse": coarse, "fine": fine or coarse, "confidence": conf, "latency_s": 2.0}


def test_evaluate_rates_bins_and_gate():
    ref = {"1": {"coarse": "paper", "fine": "paper_other"}, "2": {"coarse": "metal"},
           "3": {"coarse": "plastic"}, "4": {"coarse": "electronics"}, "5": {"coarse": "glass"}}
    pred = {"1": _p("paper", 0.95, "paper_other"), "2": _p("metal", 0.8),
            "3": _p("electronics", 0.6), "4": _p("electronics", 0.9)}
    r = ev.evaluate(ref, pred)
    assert r["coarse"] == {"n": 4, "agree": 3, "rate": 0.75}
    assert r["missing"] == 1
    assert r["conf_ge_0.7"]["rate"] == 1.0 and r["conf_ge_0.7"]["coverage"] == 0.75
    assert r["electronics"]["false_pos"] == 1
    assert r["top_confusions"] == [("plastic→electronics", 1)]
    ok, _ = ev.gate(r)
    assert ok  # 확신 ≥0.7 구간 100% ≥ 90%


def test_load_ref_formats(tmp_path):
    j = tmp_path / "a.json"
    j.write_text(json.dumps([{"id": "x", "ai_coarse": "glass", "ai_fine": "glass_etc"}]))
    assert ev.load_ref(j)["x"] == {"coarse": "glass", "fine": "glass_etc"}
    c = tmp_path / "m.csv"
    c.write_text("file,clip_top\ntaco_glass/a.jpg,glass_clear\n")
    assert ev.load_ref(c)["taco_glass/a.jpg"]["coarse"] == "glass"


def test_load_ref_jsonl_split(tmp_path):
    j = tmp_path / "v1.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "label": "glass", "label_fine": "glass_etc", "split": "gate"},
        {"id": "b", "label": "metal", "label_fine": "metal", "split": "final_holdout"}]))
    assert ev.load_ref(j, "gate") == {"a": {"coarse": "glass", "fine": "glass_etc"}}
    assert len(ev.load_ref(j)) == 2


def test_half_of_is_stable_split():
    ids = [f"id{i}" for i in range(200)]
    halves = [ev.half_of(i) for i in ids]
    assert set(halves) == {"A", "B"} and halves == [ev.half_of(i) for i in ids]


def test_stage2_splits_metal_and_derives_state():
    calls = []

    def chat(host, model, b64, prompt, schema, timeout, num_thread):
        calls.append(prompt)
        if "fine" in schema["properties"]:
            return {"fine": "metal_can"}, {}
        return {"evidence": "oil stain", "dirty": True, "broken": False, "composite": False,
                "mark": "can"}, {}

    row = {"fine": "metal", "coarse": "metal", "what": "tuna can", "confidence": 0.95}
    r, _ = lb.ask_stage2("h", "m", "b64", 1.0, row=row, chat_fn=chat)
    assert (r["fine"], r["coarse"], r["state"], r["mark_visible"]) == (
        "metal_can", "metal", "dirty", True)
    assert len(calls) == 2 and "tuna can" in calls[1]
    row = {"fine": "pet", "coarse": "plastic", "what": "bottle", "confidence": 0.9}
    calls.clear()
    r, _ = lb.ask_stage2("h", "m", "b64", 1.0, row=row, chat_fn=chat)
    assert r["fine"] == "pet" and len(calls) == 1


def test_state_priority_and_v3_fine_set():
    assert lb.state_of(dirty=True, broken=True, composite=True) == "broken"
    assert lb.state_of(dirty=True, broken=False, composite=True) == "composite"
    assert lb.state_of(dirty=False, broken=False, composite=False) == "clean"
    assert "metal" not in lb.FINE_V3 and len(lb.FINE_V3) == 27
    assert set(lb.FINE_V3) == set(lb.FINE_TO_COARSE_V3)
    assert lb.STAGE2_SHA != lb.PROMPT_SHA
