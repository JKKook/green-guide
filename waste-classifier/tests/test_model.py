"""모델 테스트 (MLP + CNN)."""
from __future__ import annotations

import pytest
import torch

from src import config
from src.model import WasteClassifierCNN, WasteClassifierMLP, build_model, count_parameters

# ───── MLP ─────

def test_mlp_forward_output_shape() -> None:
    model = WasteClassifierMLP()
    x = torch.randn(8, config.INPUT_DIM)
    y = model(x)
    assert y.shape == (8, config.NUM_CLASSES)


def test_mlp_forward_output_dtype() -> None:
    model = WasteClassifierMLP()
    x = torch.randn(2, config.INPUT_DIM)
    y = model(x)
    assert y.dtype == torch.float32


def test_mlp_count_parameters_nonzero() -> None:
    assert count_parameters(WasteClassifierMLP()) > 0


def test_mlp_invalid_dropout_length() -> None:
    with pytest.raises(ValueError):
        WasteClassifierMLP(hidden_dims=(128, 64), dropout_rates=(0.5,))


def test_mlp_backward_pass() -> None:
    model = WasteClassifierMLP()
    x = torch.randn(4, config.INPUT_DIM)
    y_true = torch.tensor([0, 1, 2, 3])
    loss = torch.nn.functional.cross_entropy(model(x), y_true)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


# ───── CNN ─────

def test_cnn_forward_output_shape() -> None:
    # pretrained=False 로 다운로드 회피 (테스트 속도 + 오프라인 호환)
    model = WasteClassifierCNN(pretrained=False)
    x = torch.randn(4, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y = model(x)
    assert y.shape == (4, config.NUM_CLASSES)
    assert y.dtype == torch.float32


def test_cnn_freeze_backbone_only_fc_trainable() -> None:
    model = WasteClassifierCNN(pretrained=False, freeze_backbone=True)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    # backbone.fc 만 학습 대상
    assert all(n.startswith("backbone.fc") for n in trainable_names), trainable_names


def test_cnn_backward_pass() -> None:
    model = WasteClassifierCNN(pretrained=False)
    x = torch.randn(2, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y_true = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y_true)
    loss.backward()
    # 적어도 fc 의 grad 는 존재해야 함
    fc_grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone.fc")]
    assert all(g is not None for g in fc_grads)


# ───── build_model ─────

@pytest.mark.parametrize("arch,expected_cls", [
    ("mlp", WasteClassifierMLP),
    ("cnn", WasteClassifierCNN),
])
def test_build_model_returns_correct_class(arch: str, expected_cls: type) -> None:
    # cnn 의 경우 pretrained 다운로드 회피를 위해 monkeypatch 없이 build_model 그대로 호출
    # 다만 빠른 테스트를 위해 CPU에서만 검증, 다운로드는 첫 1회만 발생
    model = build_model(arch)
    assert isinstance(model, expected_cls)


def test_build_model_invalid_arch() -> None:
    with pytest.raises(ValueError):
        build_model("invalid")
