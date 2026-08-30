"""Phase 7: ONNX export + 등가성 검증 (MLP/CNN 공용)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src import config
from src.model import CamWasteClassifierCNN, WasteClassifierCNN, build_model
from src.train import _model_kind


def _dummy_input(arch: str) -> torch.Tensor:
    if arch == "mlp":
        return torch.randn(1, config.INPUT_DIM, dtype=torch.float32)
    if arch in ("cnn", "cnn_edge"):
        return torch.randn(1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE,
                           dtype=torch.float32)
    raise ValueError(f"unsupported arch={arch!r}")


def _input_name(arch: str) -> str:
    return {"mlp": "vector", "cnn": "image", "cnn_edge": "edge"}[arch]


def _test_batch(arch: str, batch_size: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    if arch == "mlp":
        return rng.standard_normal((batch_size, config.INPUT_DIM)).astype(np.float32)
    if arch in ("cnn", "cnn_edge"):
        return rng.standard_normal(
            (batch_size, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE),
        ).astype(np.float32)
    raise ValueError(f"unsupported arch={arch!r}")


def export_onnx(arch: str = "mlp", opset: int = 17) -> Path:
    if arch not in config.SUPPORTED_ARCHS:
        raise ValueError(f"unsupported arch={arch!r}")

    config.ensure_directories()
    ckpt_path = config.arch_subdir(config.CHECKPOINTS_DIR, arch) / "best.pt"
    out_path = config.arch_subdir(config.MODELS_DIR, arch) / "classifier.onnx"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(_model_kind(arch))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # "cnn" (color stream) 은 (logits, cam) 둘 다 출력하도록 wrap.
    # mlp / cnn_edge 는 단일 출력 그대로.
    if arch == "cnn" and isinstance(model, WasteClassifierCNN):
        export_model: torch.nn.Module = CamWasteClassifierCNN(model)
        export_model.eval()  # wrapper + 모든 submodule eval 모드 (BN, dropout)
        # logits + cam(설명) + embedding(512d, open-set OOD reject 용)
        output_names = ["logits", "cam", "embedding"]
        dynamic_axes = {
            _input_name(arch): {0: "batch"},
            "logits": {0: "batch"},
            "cam": {0: "batch"},
            "embedding": {0: "batch"},
        }
    else:
        export_model = model
        output_names = ["logits"]
        dynamic_axes = {_input_name(arch): {0: "batch"}, "logits": {0: "batch"}}

    in_name = _input_name(arch)
    torch.onnx.export(
        export_model,
        _dummy_input(arch),
        out_path,
        input_names=[in_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    test_batch = _test_batch(arch)
    with torch.no_grad():
        torch_out_all = export_model(torch.from_numpy(test_batch))
        torch_logits = (torch_out_all[0] if isinstance(torch_out_all, tuple)
                        else torch_out_all).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {in_name: test_batch})[0]

    diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"[export:{arch}] ONNX saved → {out_path}")
    print(f"[export:{arch}] outputs: {output_names}")
    print(f"[export:{arch}] PyTorch vs ONNX (logits) max abs diff: {diff:.3e}")
    if diff > 1e-4:
        raise RuntimeError(f"ONNX 출력이 PyTorch 와 크게 다름: {diff}")
    print(f"[export:{arch}] equivalence OK (tol=1e-4)")

    # cnn 의 경우 cam + embedding 출력도 sanity check
    if arch == "cnn":
        onnx_cam, onnx_emb = session.run(["cam", "embedding"], {in_name: test_batch})
        print(f"[export:{arch}] cam shape: {onnx_cam.shape} (expected: "
              f"(batch, num_classes, 7, 7))")
        print(f"[export:{arch}] embedding shape: {onnx_emb.shape} (expected: (batch, 512))")

    return out_path


if __name__ == "__main__":
    import sys
    arch = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    export_onnx(arch=arch)
