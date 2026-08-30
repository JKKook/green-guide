"""Phase 3: 모델 정의 (MLP + CNN)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from greenguide_classifier import config


class WasteClassifierMLP(nn.Module):
    """Fully-Connected NN.
    Input: (B, 150528) flatten 벡터 → Output: (B, num_classes) logits.
    num_classes 는 config.NUM_CLASSES (manifest 에서 동적 결정).
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        hidden_dims: tuple[int, ...] = config.MLP_HIDDEN_DIMS,
        dropout_rates: tuple[float, ...] = config.MLP_DROPOUT_RATES,
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(dropout_rates):
            raise ValueError("hidden_dims 와 dropout_rates 길이가 같아야 함")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h, p in zip(hidden_dims, dropout_rates, strict=False):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WasteClassifierCNN(nn.Module):
    """ImageNet pretrained ResNet18 의 마지막 fc 만 num_classes 로 교체.

    Input: (B, 3, 224, 224) CHW float32 (ImageNet 정규화 가정) → Output: (B, num_classes) logits.
    num_classes 는 config.NUM_CLASSES (manifest 에서 동적 결정).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = config.CNN_FREEZE_BACKBONE,
    ) -> None:
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(arch: str) -> nn.Module:
    """arch 문자열로 모델 인스턴스 생성.

    num_classes 를 **호출 시점**의 config.NUM_CLASSES 로 명시 전달한다.
    (기본인자는 import 시점에 고정되므로, manifest refresh 후 늘어난 클래스 수가
     반영되지 않는 버그를 방지 — 예: 전자제품 추가 후 12클래스인데 11로 생성되던 문제.)
    """
    if arch == "mlp":
        return WasteClassifierMLP(num_classes=config.NUM_CLASSES)
    if arch == "cnn":
        return WasteClassifierCNN(num_classes=config.NUM_CLASSES)
    raise ValueError(f"unsupported arch={arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CamWasteClassifierCNN(nn.Module):
    """학습된 ResNet18 분류기를 wrap — ONNX export 시 (logits, cam, embedding) 출력.

    - cam_per_class: Zhou et al. (2016) 의 원형 Class Activation Map
        cam[b, c, h, w] = sum_k W[c, k] * features[b, k, h, w]
      ResNet18 + GAP + FC 구조에선 Grad-CAM 과 수학적으로 등가 (gradient 불필요).
      서버/앱은 cam[predicted_class, :, :] 만 가져다 ReLU + 정규화 + colormap.
    - embedding: GAP 직후 512차원 특징 벡터 (fc 직전).
      open-set OOD reject 용 — 입력이 모든 클래스 prototype 에서 멀면(임베딩 거리)
      softmax 가 확신해도 '기타/분류 불가'로 reject (etc_queue prototype 과 동일 공간).
    """

    def __init__(self, base: WasteClassifierCNN) -> None:
        super().__init__()
        self.base = base

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = self.base.backbone
        # Stem
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        # ResNet blocks — 마지막 layer4 출력이 spatial feature map
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        features = b.layer4(x)  # (B, 512, 7, 7) for 224×224 input

        # Classification head (원본과 동일)
        pooled = b.avgpool(features).flatten(1)  # (B, 512) — embedding
        logits = b.fc(pooled)                     # (B, num_classes)

        # 클래스별 CAM — fc.weight 를 1×1 conv 로 적용 (einsum 과 수학적 동일).
        # einsum 은 ONNX 양자화기가 오퍼랜드를 망가뜨려(INT8 시 shape 오류)
        # conv2d 로 교체 — 양자화 친화적이고 커널 구현도 더 빠름 (트랙 B3).
        # fc.weight (num_classes, C) → (num_classes, C, 1, 1) 커널
        cam = nn.functional.conv2d(
            features, b.fc.weight.unsqueeze(-1).unsqueeze(-1))

        return logits, cam, pooled


class WasteClassifierConvNeXt(nn.Module):
    """ConvNeXt-Tiny 백본 (정확도 스프린트 A1 — v6 실험).

    ResNet18(2015) 대비 모던 아키텍처(2022). classifier 의 마지막 Linear 만
    num_classes 로 교체, 전층 fine-tune.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.convnext_tiny(weights=weights)
        in_f = self.backbone.classifier[2].in_features  # 768
        self.backbone.classifier[2] = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CamWasteClassifierConvNeXt(nn.Module):
    """ConvNeXt 용 3-output export 래퍼 (logits, cam, embedding).

    ConvNeXt classifier = [LayerNorm2d, Flatten, Linear] 이므로
    CAM = einsum(linear.weight, LN(features)) — ResNet 판과 동일한
    GAP+Linear 가정의 확장. embedding 은 LN(features) 의 GAP (768d).
    """

    def __init__(self, trained: WasteClassifierConvNeXt) -> None:
        super().__init__()
        self.features = trained.backbone.features
        self.avgpool = trained.backbone.avgpool
        self.ln = trained.backbone.classifier[0]        # LayerNorm2d
        self.fc = trained.backbone.classifier[2]        # Linear(768, C)

    def forward(self, x: torch.Tensor):
        f = self.features(x)                 # (B, 768, H/32, W/32)
        # 정확한 logits 경로 — torchvision ConvNeXt 순서: avgpool → LN → Linear
        pooled = self.ln(self.avgpool(f)).flatten(1)     # (B, 768)
        logits = self.fc(pooled)
        # CAM 은 위치별 LN 후 fc 투영 — GAP+Linear 가정의 표준 근사
        fn = self.ln(f)
        cam = torch.einsum("ck,bkhw->bchw", self.fc.weight, fn)
        return logits, cam, pooled


def build_hier_model(num_classes: int, backbone: str = "resnet18") -> nn.Module:
    """계층 학습용 백본 팩토리 (A1 실험 — env WASTE_HIER_BACKBONE 로 선택)."""
    if backbone == "resnet18":
        return WasteClassifierCNN(num_classes=num_classes)
    if backbone == "convnext_tiny":
        return WasteClassifierConvNeXt(num_classes=num_classes)
    if backbone == "resnet50":
        # ResNet 계열 — MPS 검증 연산만 사용 (convnext 는 MPS 에서 ~40배 느림 실측)
        m = WasteClassifierCNN.__new__(WasteClassifierCNN)
        nn.Module.__init__(m)
        m.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.backbone.fc = nn.Linear(m.backbone.fc.in_features, num_classes)  # 2048
        return m
    raise ValueError(f"unknown hier backbone={backbone!r}")


def build_hier_cam_wrapper(model: nn.Module) -> nn.Module:
    """백본별 3-output export 래퍼 선택."""
    if isinstance(model, WasteClassifierConvNeXt):
        return CamWasteClassifierConvNeXt(model)
    return CamWasteClassifierCNN(model)
