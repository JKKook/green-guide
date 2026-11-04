"""분류 체계 v2 계층 학습 — 세부 27(metal·food_waste 분기) + 상태 헤드 4 (TAXONOMY_V2.md, 결정 32).

v14 경로(hier_train·greenguide_common.taxonomy·서빙 taxonomy.json)는 건드리지 않는다. 이 모듈이 v2 라벨 목록을 직접 갖는다.
- 입력: build_v2_manifest.py 의 manifest_v2.csv (path, coarse, fine_v2, state, source, rule)
  fine_v2 가 있으면 세부 감독, 없으면 대분류 롤업 감독(금속·음식물 미확정 포함). state 가 비면 상태 손실 제외.
- 분할: 기존 hier_splits.json(경로 기준, test 동결) 그대로. splits 에 없는 신규 경로(재크롭 등)는 train 에만 넣는다.
- 상태 손실: effective-number 가중(β=0.999, clean 대비 상한 10배) + broken 오버샘플(×20, WeightedRandomSampler).
- 재개: 매 epoch last.pt, GREENGUIDE_HIER_RESUME=1. 스모크: GREENGUIDE_HIER_V2_MAX_STEPS·GREENGUIDE_HIER_V2_EPOCHS.

실행: GREENGUIDE_HIER_V2_MANIFEST=<csv> python -m greenguide_classifier.hier_v2 [train|export]
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from greenguide_common.logging import get_logger
from greenguide_common.taxonomy import COARSE_LABELS, FINE_LABELS, FINE_TO_COARSE
from torch.utils.data import DataLoader, WeightedRandomSampler

from greenguide_classifier import config
from greenguide_classifier.hier_dataset import HIER_SPLITS_PATH, HierImageDataset
from greenguide_classifier.hier_train import (
    PREFETCH,
    TRAIN_WORKERS,
    VAL_PERSISTENT,
    VAL_WORKERS,
    get_hier_hyperparams,
)
from greenguide_classifier.model import build_hier_model
from greenguide_classifier.train import inverse_freq_weights, pick_device, set_seed

log = get_logger(__name__)

ARCH = "cnn_hier_v2"
SPLIT_FINE = {"metal": ("metal_can", "metal_scrap"), "food_waste": ("food_waste", "food_inedible")}
FINE_V2: tuple[str, ...] = tuple(s for f in FINE_LABELS for s in SPLIT_FINE.get(f, (f,)))
FINE_V2_TO_COARSE: dict[str, str] = {
    s: FINE_TO_COARSE[f] for f in FINE_LABELS for s in SPLIT_FINE.get(f, (f,))}
NUM_FINE = len(FINE_V2)
NUM_COARSE = len(COARSE_LABELS)
F2C = [COARSE_LABELS.index(FINE_V2_TO_COARSE[s]) for s in FINE_V2]
STATES: tuple[str, ...] = ("clean", "dirty", "broken", "composite")

BACKBONE = os.getenv("GREENGUIDE_HIER_BACKBONE", "resnet50")
LABEL_SMOOTH = float(os.getenv("GREENGUIDE_HIER_LABEL_SMOOTH", "0.0"))
RESUME = os.getenv("GREENGUIDE_HIER_RESUME", "0") == "1"
STATE_LOSS_W = float(os.getenv("GREENGUIDE_HIER_V2_STATE_W", "0.5"))
BROKEN_OVERSAMPLE = float(os.getenv("GREENGUIDE_HIER_V2_BROKEN_X", "20"))
MAX_STEPS = int(os.getenv("GREENGUIDE_HIER_V2_MAX_STEPS", "0"))      # 스모크: epoch 당 최대 배치
EPOCHS = int(os.getenv("GREENGUIDE_HIER_V2_EPOCHS", "0"))            # 0 = config 값
MANIFEST = Path(os.getenv("GREENGUIDE_HIER_V2_MANIFEST", "outputs/v2_manifest/manifest_v2.csv"))
RUN = os.getenv("GREENGUIDE_HIER_V2_RUN", ARCH)                     # 산출 폴더 이름(회차별, 예 cnn_hier_v2r2)
SELECT = os.getenv("GREENGUIDE_HIER_V2_SELECT", "acc")   # best 선정: acc(대분류+0.2·세부) | material_loss(val 세부+대분류 손실 최소, 상태 제외)
SAMPLER = os.getenv("GREENGUIDE_HIER_V2_SAMPLER", "weighted")  # weighted(파손 ×BROKEN_X 복원 추출) | shuffle(v14 와 같음)
KEEP_EPOCHS = os.getenv("GREENGUIDE_HIER_V2_KEEP_EPOCHS", "0") == "1"  # 매 에폭 가중치 보관(epoch_NN.pt)
CKPT_DIR = Path(os.getenv("GREENGUIDE_HIER_V2_CKPT_DIR", str(config.CHECKPOINTS_DIR / RUN)))
MODELS_DIR = config.MODELS_DIR / RUN
SPLITS = Path(os.getenv("GREENGUIDE_HIER_V2_SPLITS", str(HIER_SPLITS_PATH)))  # 사진 ID 그룹 분할(v2.2) 등 교체용


def build_items(manifest: Path = MANIFEST) -> list[dict[str, Any]]:
    items = []
    with manifest.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fine = r["fine_v2"]
            kind, slug = ("fine", fine) if fine else ("coarse", r["coarse"])
            items.append({
                "source_path": r["path"], "sup_kind": kind, "sup_slug": slug,
                "sup_idx": FINE_V2.index(slug) if kind == "fine" else COARSE_LABELS.index(slug),
                "state_idx": STATES.index(r["state"]) if r["state"] else -1,
            })
    return items


def split_items(items: list[dict]) -> dict[str, list[int]]:
    data = json.loads(SPLITS.read_text(encoding="utf-8"))
    where = {p: s for s in ("train", "val", "test") for p in data[s]}
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    new = skipped = 0
    for i, it in enumerate(items):
        s = where.get(it["source_path"])
        if s is None:
            # splits 밖 경로: staging 밖 재크롭(절대경로, 예 crops_v2_broken)만 train 에 넣는다.
            # 그 밖(oi4 하드네거티브 3,560 등)은 v14 와 같이 제외 — 이번 회차 의사라벨 금지(리더 11-03).
            if not Path(it["source_path"]).is_absolute():
                skipped += 1
                continue
            new += 1
            s = "train"
        out[s].append(i)
    log.info(f"[{ARCH}] splits: " + ", ".join(f"{k}={len(v):,}" for k, v in out.items())
             + f" (staging 밖 재크롭 {new} → train, splits 밖 제외 {skipped})")
    return out


class V2Dataset(HierImageDataset):
    def __getitem__(self, idx: int):
        x, is_fine, sup_idx = super().__getitem__(idx)
        return x, is_fine, sup_idx, self.items[idx]["state_idx"]


class HierV2Model(nn.Module):
    """ResNet 백본 + 세부 head(fc) + 상태 head. forward → (fine_logits, state_logits)."""

    def __init__(self, backbone: str = BACKBONE) -> None:
        super().__init__()
        self.base = build_hier_model(NUM_FINE, backbone)
        self.state_fc = nn.Linear(self.base.backbone.fc.in_features, len(STATES))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer4(b.layer3(b.layer2(b.layer1(x))))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.base.backbone.avgpool(self.features(x)).flatten(1)
        return self.base.backbone.fc(pooled), self.state_fc(pooled)


class ExportV2(nn.Module):
    """ONNX 4-output: logits(27) · cam(27,h,w) · embedding · state_logits(4)."""

    def __init__(self, m: HierV2Model) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor):
        feats = self.m.features(x)
        pooled = self.m.base.backbone.avgpool(feats).flatten(1)
        fc = self.m.base.backbone.fc
        cam = F.conv2d(feats, fc.weight.unsqueeze(-1).unsqueeze(-1))
        return fc(pooled), cam, pooled, self.m.state_fc(pooled)


def effective_number_weights(counts: Counter, n: int, beta: float = 0.999, cap: float = 10.0) -> torch.Tensor:
    w = [(1 - beta) / (1 - beta ** max(counts.get(i, 0), 1)) for i in range(n)]
    base = w[0]  # clean 기준
    w = [min(v / base, cap) for v in w]
    return torch.tensor(w, dtype=torch.float32)


class V2Loss(nn.Module):
    def __init__(self, fine_w, coarse_w, state_w) -> None:
        super().__init__()
        self.register_buffer("fine_w", fine_w)
        self.register_buffer("coarse_w", coarse_w)
        self.register_buffer("state_w", state_w)
        self.register_buffer("f2c", torch.tensor(F2C, dtype=torch.long))

    def coarse_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        m = logp.max(dim=1, keepdim=True).values
        summed = torch.zeros(logp.size(0), NUM_COARSE, dtype=logp.dtype, device=logp.device
                             ).scatter_add_(1, self.f2c.expand(logp.size(0), -1), (logp - m).exp())
        return summed.clamp_min(1e-12).log() + m

    def forward(self, out, is_fine, sup_idx, state_idx):
        logits, slog = out
        losses = torch.zeros(logits.size(0), dtype=logits.dtype, device=logits.device)
        fm = is_fine.bool()
        if fm.any():
            losses[fm] = F.cross_entropy(logits[fm], sup_idx[fm], weight=self.fine_w, reduction="none",
                                         label_smoothing=LABEL_SMOOTH)
        cm = ~fm
        if cm.any():
            ci = sup_idx[cm]
            losses[cm] = -self.coarse_log_probs(logits[cm]).gather(1, ci.unsqueeze(1)).squeeze(1) * self.coarse_w[ci]
        loss = losses.mean()
        self.last_material = float(loss.detach())   # 재질(세부+대분류 롤업) 손실 — best 선정용(상태 제외)
        sm = state_idx >= 0
        if sm.any():
            loss = loss + STATE_LOSS_W * F.cross_entropy(slog[sm], state_idx[sm], weight=self.state_w)
        return loss


def class_weights(train: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fc, cc = Counter(), Counter()
    for it in train:
        if it["sup_kind"] == "fine":
            fc[it["sup_idx"]] += 1
        else:
            cc[it["sup_idx"]] += 1
            kids = [i for i, c in enumerate(F2C) if c == it["sup_idx"]]
            for k in kids:
                fc[k] += 1 / len(kids)
    cap = config.CNN_CLASS_WEIGHT_CAP
    fine_w = inverse_freq_weights({k: max(1, round(v)) for k, v in fc.items()}, NUM_FINE, cap)
    coarse_w = inverse_freq_weights(dict(cc), NUM_COARSE, cap)
    sc = Counter(it["state_idx"] for it in train if it["state_idx"] >= 0)
    return fine_w, coarse_w, effective_number_weights(sc, len(STATES))


def run_epoch(model, loader, crit, device, opt=None, desc="") -> dict:
    model.train(opt is not None)
    f2c = torch.tensor(F2C, device=device)
    tot = mat = n = fc_ok = fc_n = cc_ok = 0
    st = Counter()   # dirty tp/fp/fn, 상태 정답 수
    with torch.enable_grad() if opt else torch.no_grad():
        for step, batch in enumerate(loader):
            if MAX_STEPS and step >= MAX_STEPS:
                break
            x, is_fine, sup, sidx = (t.to(device, non_blocking=True) for t in batch)
            out = model(x)
            loss = crit(out, is_fine, sup, sidx)
            if opt:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            b = x.size(0)
            tot += loss.item() * b
            mat += crit.last_material * b
            n += b
            pf = out[0].argmax(1)
            fm = is_fine.bool()
            fc_ok += int((pf[fm] == sup[fm]).sum())
            fc_n += int(fm.sum())
            true_c = torch.where(fm, f2c[sup.clamp(max=NUM_FINE - 1)], sup)
            cc_ok += int((f2c[pf] == true_c).sum())
            sm = sidx >= 0
            ps = out[1].argmax(1)
            st["n"] += int(sm.sum())
            st["ok"] += int((ps[sm] == sidx[sm]).sum())
            d = STATES.index("dirty")
            st["tp"] += int(((ps == d) & (sidx == d)).sum())
            st["fp"] += int(((ps == d) & sm & (sidx != d)).sum())
            st["fn"] += int(((ps != d) & (sidx == d)).sum())
    return {"loss": tot / max(n, 1), "material_loss": mat / max(n, 1), "fine_acc": fc_ok / max(fc_n, 1), "coarse_acc": cc_ok / max(n, 1),
            "state_acc": st["ok"] / max(st["n"], 1),
            "dirty_recall": st["tp"] / max(st["tp"] + st["fn"], 1),
            "dirty_precision": st["tp"] / max(st["tp"] + st["fp"], 1)}


def _meta() -> dict:
    return {"arch": ARCH, "backbone": BACKBONE, "label_smooth": LABEL_SMOOTH, "fine_labels": list(FINE_V2),
            "states": list(STATES), "state_loss_w": STATE_LOSS_W, "broken_x": BROKEN_OVERSAMPLE,
            "manifest": str(MANIFEST), "select": SELECT, "splits": str(SPLITS),
            "sampler": SAMPLER}


def train() -> Path:
    config.ensure_directories()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = pick_device()
    hp = get_hier_hyperparams()
    epochs = EPOCHS or hp.num_epochs
    items = build_items()
    sp = split_items(items)
    tr, va = [items[i] for i in sp["train"]], [items[i] for i in sp["val"]]
    log.info(f"[{ARCH}] fine={NUM_FINE} states={STATES} 상태 라벨(train): "
             f"{dict(Counter(STATES[i['state_idx']] if i['state_idx'] >= 0 else '미상' for i in tr))}")
    if SAMPLER == "shuffle":
        sampler, shuffle = None, True
    else:
        sw = [BROKEN_OVERSAMPLE if it["state_idx"] == STATES.index("broken") else 1.0 for it in tr]
        sampler, shuffle = WeightedRandomSampler(sw, num_samples=len(tr), replacement=True), False
    tl = DataLoader(V2Dataset(tr, augment=True), batch_size=hp.batch_size, sampler=sampler, shuffle=shuffle,
                    num_workers=TRAIN_WORKERS, persistent_workers=TRAIN_WORKERS > 0,
                    prefetch_factor=PREFETCH if TRAIN_WORKERS else None)
    vl = DataLoader(V2Dataset(va, augment=False), batch_size=hp.batch_size, shuffle=False,
                    num_workers=VAL_WORKERS, persistent_workers=VAL_PERSISTENT and VAL_WORKERS > 0,
                    prefetch_factor=PREFETCH if VAL_WORKERS else None)
    model = HierV2Model().to(device)
    fw, cw, sw_ = class_weights(tr)
    log.info(f"[{ARCH}] 상태 가중 {dict(zip(STATES, [round(v, 2) for v in sw_.tolist()], strict=True))}")
    crit = V2Loss(fw, cw, sw_).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

    last, best = CKPT_DIR / "last.pt", CKPT_DIR / "best.pt"
    hist, best_score, best_epoch, patience, start = [], float("-inf"), -1, 0, 1
    if RESUME and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if {k: ck.get(k, "acc" if k == "select" else None) for k in ("backbone", "fine_labels", "states", "select")} == \
                {k: _meta()[k] for k in ("backbone", "fine_labels", "states", "select")}:
            model.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["optimizer_state"])
            hist, best_score, best_epoch = ck["history"], ck["best_score"], ck["best_epoch"]
            patience, start = ck["patience"], ck["epoch"] + 1
            log.info(f"[{ARCH}] last.pt 재개 — epoch {start} 부터 (best {best_epoch}, {best_score:.4f})")
        else:
            log.warning(f"[{ARCH}] last.pt 설정 불일치 — 처음부터")

    for ep in range(start, epochs + 1):
        t0 = time.time()
        a = run_epoch(model, tl, crit, device, opt)
        v = run_epoch(model, vl, crit, device)
        score = -v["material_loss"] if SELECT == "material_loss" else v["coarse_acc"] + 0.2 * v["fine_acc"]
        hist.append({"epoch": ep, "train": a, "val": v, "sec": round(time.time() - t0, 1)})
        log.info(f"[{ARCH} epoch {ep}] train {a} | val {v} | {time.time() - t0:.0f}s")
        if score > best_score:
            best_score, best_epoch, patience = score, ep, 0
            torch.save(_meta() | {"epoch": ep, "model_state": model.state_dict(), "val": v}, best)
        else:
            patience += 1
        if KEEP_EPOCHS:
            torch.save(_meta() | {"epoch": ep, "model_state": model.state_dict(), "val": v}, CKPT_DIR / f"epoch_{ep:02d}.pt")
        tmp = last.with_suffix(".pt.tmp")
        torch.save(_meta() | {"epoch": ep, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                              "history": hist, "best_score": best_score, "best_epoch": best_epoch,
                              "patience": patience}, tmp)
        tmp.replace(last)
        if patience >= hp.patience:
            log.info(f"[{ARCH}] early stopping at epoch {ep}")
            break
    (CKPT_DIR / "training_log.json").write_text(json.dumps(
        _meta() | {"best_epoch": best_epoch, "best_score": best_score, "history": hist},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"[{ARCH}] done. best epoch={best_epoch} → {best}")
    return best


def calibrate_state_gate(m: HierV2Model, target_acc: float = 0.9) -> dict:
    """val 의 상태 라벨 있는 장에서 τ 를 고른다 — 최대 확률 ≥τ 인 예측의 정확도가 target 이상이 되는 가장 작은 τ.

    표본은 GREENGUIDE_HIER_V2_GATE_N(기본 3000)장. 근거(정확도·커버리지·dirty 정밀도)를 함께 남긴다.
    """
    n = int(os.getenv("GREENGUIDE_HIER_V2_GATE_N", "3000"))
    items = build_items()
    import random  # noqa: PLC0415
    pool = [items[i] for i in split_items(items)["val"] if items[i]["state_idx"] >= 0]
    val = random.Random(0).sample(pool, min(n, len(pool)))   # 경로 정렬 편향 방지(앞쪽 n 장은 clean 위주)
    device = pick_device()
    m = m.to(device).eval()
    probs, ys = [], []
    loader = DataLoader(V2Dataset(val, augment=False), batch_size=64, num_workers=VAL_WORKERS)
    with torch.no_grad():
        for x, _, _, s in loader:
            probs.append(F.softmax(m(x.to(device))[1], 1).cpu())
            ys.append(s)
    p, y = torch.cat(probs), torch.cat(ys)
    conf, pred = p.max(1)
    d = STATES.index("dirty")
    best = None
    for t in [round(0.30 + 0.05 * k, 2) for k in range(14)]:
        keep = conf >= t
        if keep.sum() == 0:
            break
        acc = float((pred[keep] == y[keep]).float().mean())
        pd_ = keep & (pred == d)
        row = {"tau": t, "acc": round(acc, 4), "coverage": round(float(keep.float().mean()), 4),
               "dirty_precision": round(float((y[pd_] == d).float().mean()), 4) if pd_.any() else None,
               "n": int(len(y)), "target_acc": target_acc}
        if acc >= target_acc:
            best = row
            break
        best = row
    m.cpu()
    log.info(f"[{ARCH}] state_gate 보정: {best}")
    return best


def export(opset: int = 17) -> Path:
    import numpy as np  # noqa: PLC0415
    import onnx  # noqa: PLC0415

    from greenguide_classifier.hier_export import GATE_DEFAULTS  # noqa: PLC0415
    from greenguide_classifier.infer import load_session  # noqa: PLC0415

    ck = torch.load(CKPT_DIR / "best.pt", map_location="cpu", weights_only=False)
    if tuple(ck["fine_labels"]) != FINE_V2 or tuple(ck["states"]) != STATES:
        raise RuntimeError("체크포인트 라벨과 v2 목록 불일치")
    m = HierV2Model(ck["backbone"])
    m.load_state_dict(ck["model_state"])
    em = ExportV2(m.eval()).eval()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "classifier.onnx"
    torch.onnx.export(em, torch.randn(1, 3, 224, 224), out, input_names=["image"],
                      output_names=["logits", "cam", "embedding", "state_logits"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"},
                                    "cam": {0: "batch", 2: "cam_h", 3: "cam_w"}, "embedding": {0: "batch"},
                                    "state_logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(out))
    x = np.random.default_rng(0).standard_normal((2, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        tl, _, _, ts = em(torch.from_numpy(x))
    ol, _, _, os_ = load_session(out).run(["logits", "cam", "embedding", "state_logits"], {"image": x})
    diff = max(float(abs(tl.numpy() - ol).max()), float(abs(ts.numpy() - os_).max()))
    if diff > 1e-4:
        raise RuntimeError(f"ONNX 출력 불일치 {diff}")
    assert os_.shape == (2, len(STATES)) and os_.dtype == np.float32, os_.shape
    gate = calibrate_state_gate(m)
    (MODELS_DIR / "taxonomy.json").write_text(json.dumps({
        "version": "hier_v2_taxonomy_v2", "backbone": ck["backbone"], "fine_labels": list(FINE_V2),
        "coarse_labels": list(COARSE_LABELS), "fine_to_coarse": FINE_V2_TO_COARSE,
        "fine_idx_to_coarse_idx": F2C, "gate": GATE_DEFAULTS,
        # GG_API 계약(11-03): state_labels = state_logits 순서, state_gate.min_confidence = val 보정값
        "state_labels": list(STATES), "state_gate": {"min_confidence": gate["tau"] if gate else 1.01}, "state_gate_basis": gate,  # 보정 불가(STATE_W=0)면 상태 미출력
        "best_epoch": ck["epoch"], "val": ck.get("val")}, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[{ARCH}] ONNX → {out} (diff {diff:.1e}), taxonomy.json 동봉")
    return out


if __name__ == "__main__":
    {"train": train, "export": export}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
