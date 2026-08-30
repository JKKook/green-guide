"""계층 모델 ONNX export — 기존 3-output 컨벤션 + taxonomy 사이드카.

산출물 (outputs/models/cnn_hier/):
- classifier.onnx : 입력 "image" (B,3,224,224) → logits(B,25) + cam(B,25,7,7) + embedding(B,512)
- taxonomy.json   : fine/coarse 라벨, fine→coarse 매핑, 신뢰도 게이트 임계
                    → waste-api 가 DB 없이도 롤업·게이트 가능

실행: .venv/bin/python -m src.hier_export
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src import config
from src.hier_train import CKPT_DIR
from src.model import build_hier_cam_wrapper, build_hier_model
from src.taxonomy import (
    COARSE_LABELS, FINE_IDX_TO_COARSE_IDX, FINE_LABELS, FINE_TO_COARSE, NUM_FINE,
)

MODELS_DIR = config.MODELS_DIR / "cnn_hier"

# 신뢰도 게이트 기본값 (blueprint §2.3) — 서빙에서 override 가능
GATE_DEFAULTS = {
    "fine_min_confidence": 0.60,   # 세부 표시 최소 확신
    "fine_min_margin": 0.15,       # 세부 top1-top2 최소 격차
    "coarse_min_confidence": 0.55, # 대분류 표시 최소 확신 (미달 → reject/etc)
}


def export_hier_onnx(opset: int = 17) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "best.pt"
    out_path = MODELS_DIR / "classifier.onnx"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 체크포인트의 라벨과 현재 taxonomy 일치 검증 (드리프트 방지)
    ckpt_fine = ckpt.get("fine_labels")
    if ckpt_fine and tuple(ckpt_fine) != tuple(FINE_LABELS):
        raise RuntimeError(
            "체크포인트 fine_labels 와 현재 taxonomy 불일치 — "
            f"ckpt={len(ckpt_fine)}개 vs taxonomy={len(FINE_LABELS)}개",
        )

    backbone = ckpt.get("backbone", "resnet18")
    model = build_hier_model(NUM_FINE, backbone)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    export_model = build_hier_cam_wrapper(model)
    export_model.eval()

    dummy = torch.randn(1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    # H/W 동적: ResNet 은 GAP 까지 fully-convolutional 이라 448² 입력 시
    # CAM 이 (C,14,14) 로 — 고해상 재질 맵 용 (CAM_MATERIAL_UPGRADE_PLAN Stage 1-1).
    torch.onnx.export(
        export_model, dummy, out_path,
        input_names=["image"],
        output_names=["logits", "cam", "embedding"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
            "cam": {0: "batch", 2: "cam_h", 3: "cam_w"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(out_path))

    # 등가성 검증 (기존 export.py 와 동일 규율)
    rng = np.random.default_rng(0)
    batch = rng.standard_normal(
        (4, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE),
    ).astype(np.float32)
    with torch.no_grad():
        torch_logits = export_model(torch.from_numpy(batch))[0].numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_cam, onnx_emb = sess.run(
        ["logits", "cam", "embedding"], {"image": batch},
    )
    diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"[export:cnn_hier] ONNX → {out_path}")
    print(f"[export:cnn_hier] logits max abs diff: {diff:.3e}")
    if diff > 1e-4:
        raise RuntimeError(f"ONNX 출력 불일치: {diff}")
    assert onnx_cam.shape == (4, NUM_FINE, 7, 7), onnx_cam.shape
    assert onnx_emb.shape[0] == 4 and onnx_emb.shape[1] in (512, 768, 2048), onnx_emb.shape
    print(f"[export:cnn_hier] equivalence OK, cam {onnx_cam.shape}, emb {onnx_emb.shape}")

    # 동적 해상도 sanity — 448² 입력 시 CAM (1, C, 14, 14) (고해상 재질 맵)
    hi = rng.standard_normal((1, 3, 448, 448)).astype(np.float32)
    _, cam_hi, _ = sess.run(["logits", "cam", "embedding"], {"image": hi})
    assert cam_hi.shape == (1, NUM_FINE, 14, 14), f"hi-res cam {cam_hi.shape}"
    print(f"[export:cnn_hier] hi-res cam OK: {cam_hi.shape} (448² 입력)")

    # taxonomy 사이드카 — 서빙이 DB 없이 롤업/게이트 수행하는 근거
    sidecar = {
        "version": "hier_v2",
        "backbone": backbone,
        "fine_labels": list(FINE_LABELS),
        "coarse_labels": list(COARSE_LABELS),
        "fine_to_coarse": dict(FINE_TO_COARSE),
        "fine_idx_to_coarse_idx": list(FINE_IDX_TO_COARSE_IDX),
        "gate": GATE_DEFAULTS,
        "best_epoch": ckpt.get("epoch"),
        "val_fine_acc": ckpt.get("val_fine_acc"),
        "val_coarse_acc": ckpt.get("val_coarse_acc"),
    }
    sidecar_path = MODELS_DIR / "taxonomy.json"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[export:cnn_hier] taxonomy sidecar → {sidecar_path}")
    return out_path


if __name__ == "__main__":
    export_hier_onnx()
