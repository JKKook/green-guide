"""Grad-CAM 시각화 — "모델이 어느 영역을 보고 이 클래스로 판단했는가".

ResNet18 의 마지막 conv block (layer4) 의 activation × gradient 을 이용해
입력 이미지의 어느 부분이 예측에 가장 기여했는지 heatmap 으로 출력.

흐름:
  1. 학습된 best.pt 로드
  2. 입력 이미지(들) 을 ImageNet 정규화 → 모델에 입력
  3. forward 시 layer4 activation 캡처, backward 시 gradient 캡처
  4. Grad-CAM = ReLU( sum_k ( mean_hw(gradient[k]) * activation[k] ) )
  5. 7×7 → 224×224 resize → jet colormap → 원본 이미지에 alpha-blend
  6. side-by-side (원본 / overlay) PNG 저장

사용:
    cd greenguide-classifier
    .venv/bin/python visualize_cam.py --image <path>
    .venv/bin/python visualize_cam.py --label plastic --n 5   # 클래스에서 5장 샘플
    .venv/bin/python visualize_cam.py --image x.jpg --target-class trash  # 강제 클래스
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from greenguide_common import imaging, settings
from greenguide_common.logging import get_logger

from greenguide_classifier import config
from greenguide_classifier.model import WasteClassifierCNN

log = get_logger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent
CKPT_PATH: Path = PROJECT_ROOT / "outputs" / "checkpoints" / "cnn" / "best.pt"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "cam"


# ImageNet 정규화 (train 코드와 동일해야 함)
_NORMALIZE = transforms.Normalize(
    mean=list(imaging.IMAGENET_MEAN),
    std=list(imaging.IMAGENET_STD),
)
_PREPROCESS = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    _NORMALIZE,
])


class CamGenerator:
    """ResNet18 의 layer4 에 hook 을 걸고 Grad-CAM 생성."""

    def __init__(self, model: WasteClassifierCNN, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

        # ResNet18 의 layer4 — 마지막 conv block, output (B, 512, 7, 7)
        self.target_layer = model.backbone.layer4

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generate(
        self, image_tensor: torch.Tensor, target_class: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """이미지 1장 → (cam[H, W] in [0,1], target_class, probs[C])."""
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Forward
        logits = self.model(image_tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward — target_class 의 logit 에 대해서만
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        # Grad-CAM 계산
        acts = self._activations[0]   # (512, 7, 7)
        grads = self._gradients[0]    # (512, 7, 7)
        # 각 채널의 중요도 = gradient 의 global average (Grad-CAM 공식)
        weights = grads.mean(dim=(1, 2))   # (512,)
        cam = torch.einsum("c,chw->hw", weights, acts)  # (7, 7)
        cam = F.relu(cam)  # 음수는 "이 클래스와 반대 방향" → 0 으로 clip
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min).abs() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.cpu().numpy(), target_class, probs


def _load_image(path: Path) -> tuple[torch.Tensor, Image.Image]:
    """이미지 path → (tensor for model, original PIL for display)."""
    pil = Image.open(path).convert("RGB")
    tensor = _PREPROCESS(pil)
    return tensor, pil


def _overlay(orig_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """7×7 CAM → 원본 사이즈로 resize → jet colormap → 원본과 alpha-blend."""
    orig_arr = np.array(orig_pil)
    h, w = orig_arr.shape[:2]
    # 7×7 → 원본 사이즈 (PIL 로 bilinear resize)
    cam_pil = Image.fromarray(np.uint8(cam * 255))
    cam_resized = np.array(cam_pil.resize((w, h), Image.BILINEAR)) / 255.0
    # matplotlib jet colormap → RGB
    cmap = plt.get_cmap("jet")
    heatmap_rgba = cmap(cam_resized)
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)
    # alpha blend
    overlay = (alpha * heatmap_rgb + (1 - alpha) * orig_arr).astype(np.uint8)
    return Image.fromarray(overlay)


def visualize(
    image_path: Path,
    output_path: Path,
    cam_gen: CamGenerator,
    target_class: int | None = None,
) -> tuple[str, float]:
    """1장 시각화 → output_path 에 PNG 저장. Returns: (predicted_label, confidence)."""
    tensor, orig_pil = _load_image(image_path)
    cam, target_idx, probs = cam_gen.generate(tensor, target_class=target_class)
    overlay_img = _overlay(orig_pil, cam, alpha=0.45)

    label = config.CLASS_LABELS[target_idx]
    confidence = float(probs[target_idx])

    # side-by-side render
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(orig_pil)
    axes[0].set_title(f"input: {image_path.name}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(overlay_img)
    explained = "predicted" if target_class is None else f"forced → {label}"
    axes[1].set_title(
        f"{label} ({confidence * 100:.1f}%) · {explained}\n"
        + " · ".join(
            f"{config.CLASS_LABELS[i]}={probs[i]*100:.0f}%"
            for i in np.argsort(probs)[::-1][:3]
        ),
        fontsize=9,
    )
    axes[1].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return label, confidence


def _sample_images_for_label(label: str, n: int) -> list[Path]:
    """greenguide-preprocessor 의 raw 폴더에서 해당 라벨 이미지 n 장 샘플."""
    raw_dir = (
        settings.PREPROCESSOR_ROOT
        / "data" / "raw" / "garbage-classification" / label
    )
    if not raw_dir.exists():
        sys.exit(f"ERROR: {raw_dir} 가 없습니다. 라벨이 올바른지 확인하세요.")
    images = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not images:
        sys.exit(f"ERROR: {raw_dir} 에 이미지가 없습니다.")
    # 균등 샘플 (앞·중간·뒤)
    step = max(1, len(images) // n)
    return images[::step][:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM 시각화")
    src = parser.add_mutually_exclusive_group(required=True)
    greenguide_classifier.add_argument("--image", type=Path, help="시각화할 단일 이미지 경로")
    greenguide_classifier.add_argument("--label", type=str, help="이 라벨 폴더에서 자동 샘플")
    parser.add_argument("--n", type=int, default=5,
                        help="--label 사용 시 샘플 개수 (default 5)")
    parser.add_argument("--target-class", type=str, default=None,
                        help=f"강제 target 클래스 (default = predicted). "
                             f"가능: {list(config.CLASS_LABELS)}")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"출력 폴더 (default: {OUTPUT_DIR})")
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH,
                        help=f"checkpoint 경로 (default: {CKPT_PATH})")
    args = parser.parse_args()

    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"device: {device}")

    # 체크포인트 로드
    if not args.ckpt.exists():
        sys.exit(f"ERROR: 체크포인트 없음 → {args.ckpt}\n"
                 "  먼저 `python main.py all --arch cnn` 으로 학습하세요.")
    model = WasteClassifierCNN(pretrained=False)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    # 체크포인트 형식 다양성 대응
    if isinstance(state, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                model.load_state_dict(state[key])
                break
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    log.info(f"loaded: {args.ckpt}")

    # target_class 검증 + 변환
    target_idx: int | None = None
    if args.target_class is not None:
        if args.target_class not in config.CLASS_LABELS:
            sys.exit(f"ERROR: --target-class={args.target_class!r} 는 "
                     f"유효한 클래스가 아닙니다. 가능: {list(config.CLASS_LABELS)}")
        target_idx = config.CLASS_LABELS.index(args.target_class)

    # 이미지 목록 결정
    if args.image:
        if not args.image.exists():
            sys.exit(f"ERROR: 이미지 없음 → {args.image}")
        images = [args.image]
    else:
        images = _sample_images_for_label(args.label, args.n)
        log.info(f"sampled {len(images)} images from label={args.label!r}")

    # 실행
    with CamGenerator(model, device) as gen:
        for img_path in images:
            out_path = args.output_dir / f"cam_{img_path.stem}.png"
            label, conf = visualize(
                img_path, out_path, gen, target_class=target_idx,
            )
            print(f"  ✓ {img_path.name} → {label} ({conf*100:.1f}%) "
                  f"→ {out_path.relative_to(PROJECT_ROOT)}")

    print(f"\n✓ {len(images)}장 완료 — {args.output_dir.relative_to(PROJECT_ROOT)}/ 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
