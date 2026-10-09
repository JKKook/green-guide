#!/usr/bin/env python3
"""DINOv2 계층(25-fine) 선형 헤드 — hier 앙상블용 (정확도 스프린트 A2).

배경: 서버의 DINOv2 앙상블은 구 13클래스 flat 이라 /predict-hier 에서 미사용.
DINOv2 임베딩은 동결이므로 25클래스 선형 헤드만 재학습하면 계층 경로에
검증된 confident-wrong 보정을 복원할 수 있다 (비용 대비 최고 효율 레버).

흐름 (build_dinov2_classifier.py 패턴의 계층판):
  1. hier fine-감독 train/val 아이템에서 DINOv2-small CLS 임베딩 추출
     (클래스당 CAP 서브샘플 — 선형 헤드엔 충분)
  2. Linear(384→25) 학습 (fine CE, 클래스 가중치)
  3. 통합 ONNX (image → fine logits) + labels 사이드카 export
  4. frozen test 로 단독/앙상블 성능 리포트

사용: .venv/bin/python scripts/build_dinov2_hier_head.py [--skip-extract]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict

import _base  # noqa: F401 — sys.path 설정

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


import numpy as np
import torch
import torch.nn as nn
from greenguide_common import imaging
from greenguide_common.taxonomy import FINE_LABELS, NUM_FINE
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from greenguide_classifier import config
from greenguide_classifier.hier_dataset import build_hier_items, load_or_build_hier_splits
from greenguide_classifier.infer import load_session, pick_device, softmax

ImageFile.LOAD_TRUNCATED_IMAGES = True

DINOV2_NAME = "facebook/dinov2-small"
EMBED_DIM = 384
INPUT_SIZE = 224
PER_CLASS_CAP = 3000        # 선형 헤드용 서브샘플 상한
OUT_DIR = config.MODELS_DIR / "dinov2_hier"
SEED = 42


class _ImgSet(Dataset):
    """DINOv2 전용 전처리 (ImageNet 정규화, 224²)."""

    MEAN = torch.tensor(list(imaging.IMAGENET_MEAN)).view(3, 1, 1)
    STD = torch.tensor(list(imaging.IMAGENET_STD)).view(3, 1, 1)

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        p = config.PREPROCESSOR_ROOT / it["source_path"]
        with Image.open(p) as im:
            im = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        x = torch.from_numpy(
            np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return (x - self.MEAN) / self.STD, it["sup_idx"]


def _subsample(items, splits, split_name, rng):
    by_c = defaultdict(list)
    for i in splits[split_name]:
        it = items[i]
        if it["sup_kind"] == "fine":
            by_c[it["sup_idx"]].append(it)
    out = []
    for _c, pool in by_c.items():
        rng.shuffle(pool)
        out.extend(pool[:PER_CLASS_CAP])
    return out


def extract(split_items, tag: str, model, device) -> tuple[np.ndarray, np.ndarray]:
    cache = OUT_DIR / f"emb_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"[{tag}] 캐시 사용: {d['x'].shape}")
        return d["x"], d["y"]
    loader = DataLoader(_ImgSet(split_items), batch_size=64,
                        num_workers=6, persistent_workers=True)
    xs, ys = [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            out = model(pixel_values=x.to(device))
            cls = out.last_hidden_state[:, 0]          # CLS token (B, 384)
            xs.append(cls.cpu().numpy())
            ys.append(y.numpy())
            if bi % 50 == 0:
                done = (bi + 1) * 64
                print(f"[{tag}] {done:,}/{len(split_items):,} "
                      f"({(time.time()-t0):.0f}s)", flush=True)
    X = np.concatenate(xs).astype(np.float32)
    Y = np.concatenate(ys).astype(np.int64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=X, y=Y)
    print(f"[{tag}] 추출 완료 {X.shape} → {cache}")
    return X, Y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.parse_args()

    import random
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    from transformers import AutoModel
    device = pick_device()
    print(f"device={device}")
    # eager attention — torch 2.4 의 SDPA ONNX export 버그 회피
    dinov2 = AutoModel.from_pretrained(
        DINOV2_NAME, attn_implementation="eager").eval().to(device)

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    tr = _subsample(items, splits, "train", rng)
    va = _subsample(items, splits, "val", rng)
    te = [items[i] for i in splits["test"] if items[i]["sup_kind"] == "fine"]
    print(f"train {len(tr):,} / val {len(va):,} / test(fine) {len(te):,}")

    Xtr, Ytr = extract(tr, "train", dinov2, device)
    Xva, Yva = extract(va, "val", dinov2, device)
    Xte, Yte = extract(te, "test", dinov2, device)

    # ── 선형 헤드 학습 ──
    cnt = Counter(Ytr.tolist())
    total = len(Ytr)
    raw = [total / (NUM_FINE * cnt.get(i, 1)) for i in range(NUM_FINE)]
    med = statistics.median(raw)
    w = torch.tensor([min(v, med * 4) for v in raw], dtype=torch.float32)

    head = nn.Linear(EMBED_DIM, NUM_FINE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xv, Yv = torch.from_numpy(Xva), torch.from_numpy(Yva)

    best_acc, best_state = 0.0, None
    for ep in range(1, 31):
        head.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            loss = crit(head(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xv).argmax(1) == Yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0:
            print(f"ep{ep:02d} val_acc={acc:.4f} (best {best_acc:.4f})")
    head.load_state_dict(best_state)

    # ── test 리포트: 단독 + ResNet 앙상블 ──
    with torch.no_grad():
        dino_logits = head(torch.from_numpy(Xte)).numpy()
    dino_acc = float((dino_logits.argmax(1) == Yte).mean())
    print(f"\nDINOv2 단독 fine acc(test): {dino_acc:.4f}")

    sess = load_session(config.MODELS_DIR / "cnn_hier" / "classifier.onnx")
    ld = DataLoader(_ImgSet(te), batch_size=64, num_workers=6)
    res_logits = []
    for x, _ in ld:
        (lg,) = sess.run(["logits"], {"image": x.numpy()})
        res_logits.append(lg)
    R = np.concatenate(res_logits)
    res_acc = float((R.argmax(1) == Yte).mean())
    for wd in (0.3, 0.4, 0.5, 0.6):
        ens = (1 - wd) * softmax(R, axis=1) + wd * softmax(dino_logits, axis=1)
        acc = float((ens.argmax(1) == Yte).mean())
        print(f"ResNet {res_acc:.4f} | 앙상블(w_dino={wd}): {acc:.4f}")

    # ── 통합 ONNX export ──
    class Combined(nn.Module):
        def __init__(self, backbone, lin):
            super().__init__()
            self.backbone, self.lin = backbone, lin

        def forward(self, image):
            out = self.backbone(pixel_values=image)
            return self.lin(out.last_hidden_state[:, 0])

    comb = Combined(dinov2.cpu().eval(), head.eval())
    onnx_path = OUT_DIR / "dinov2_hier.onnx"
    torch.onnx.export(
        comb, torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE), onnx_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    (OUT_DIR / "labels.json").write_text(
        json.dumps(list(FINE_LABELS), ensure_ascii=False), encoding="utf-8")

    # 등가성 검증
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        t_out = comb(x).numpy()
    s = load_session(onnx_path)
    (o_out,) = s.run(["logits"], {"image": x.numpy()})
    diff = float(np.abs(t_out - o_out).max())
    print(f"\nONNX export → {onnx_path} (diff {diff:.2e})")
    assert diff < 1e-3
    print("DINOV2_HIER_DONE")


if __name__ == "__main__":
    main()
