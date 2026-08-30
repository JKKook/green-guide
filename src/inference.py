"""ONNX 모델 추론 wrapper — single 또는 ensemble (color + edge)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """(B, C) logits → 확률 (numerically stable)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class WasteClassifier:
    """Color 모델 단일 / Color+Edge ensemble 모두 지원.

    - edge_model_path 가 주어지면 ensemble 모드로 작동
    - predict_color() 는 color 입력 받음
    - predict_ensemble() 는 (color, edge) 입력 받음
    - predict() 는 사용 가능한 모델 따라 자동 선택
    - labels: ONNX output 순서와 일치하는 클래스 라벨. 동적 N개 지원 (active
      model_versions row 의 class_labels 가 정본; 없으면 config.CLASS_LABELS).
    """

    def __init__(
        self,
        model_path: Path | None = None,
        edge_model_path: Path | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.model_path = model_path or config.MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX 모델 파일을 찾을 수 없음: {self.model_path}\n"
                f"waste-classifier 의 학습·export 를 먼저 완료해주세요."
            )

        self.arch = config.get_model_arch()
        self.input_name = config.get_input_name(self.arch)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        # 라벨 — 동적 (remote meta) 우선, 없으면 정적 fallback
        self.labels: tuple[str, ...] = tuple(labels) if labels else config.CLASS_LABELS

        # Edge stream (선택)
        self.edge_model_path = edge_model_path or config.EDGE_MODEL_PATH
        self.edge_session: ort.InferenceSession | None = None
        self.edge_input_name = "edge"
        if self.edge_model_path and self.edge_model_path.exists():
            self.edge_session = ort.InferenceSession(
                str(self.edge_model_path),
                providers=["CPUExecutionProvider"],
            )

    @property
    def has_edge_stream(self) -> bool:
        return self.edge_session is not None

    @property
    def has_cam_output(self) -> bool:
        """color 모델이 (logits, cam) 2-output 으로 export 됐는지."""
        return "cam" in {o.name for o in self.session.get_outputs()}

    def _run(self, session: ort.InferenceSession, input_name: str,
             tensor: np.ndarray) -> np.ndarray:
        return session.run([config.ONNX_OUTPUT_NAME], {input_name: tensor})[0]

    def region_cam(self, color_input: np.ndarray) -> tuple[dict, np.ndarray | None]:
        """분류 결과 + 전체 클래스 CAM (C, h, w) 반환 — 다중재질 영역 분석용.
        cam-aware ONNX 가 아니면 cam=None.
        """
        result = self.predict(color_input)
        _, cam_all = self._run_color_with_cam(color_input)
        cam = cam_all[0] if cam_all is not None else None  # (C, h, w)
        return result, cam

    def _run_color_with_cam(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """color 모델 1회 호출 → (logits, cam_per_class or None).

        cam_per_class shape: (B, num_classes, H, W) — 보통 (1, 6, 7, 7).
        모델이 단일 출력이면 cam=None.
        """
        if self.has_cam_output:
            logits, cam = self.session.run(
                ["logits", "cam"], {self.input_name: tensor},
            )
            return logits, cam
        logits = self.session.run(["logits"], {self.input_name: tensor})[0]
        return logits, None

    def predict(
        self,
        color_input: np.ndarray,
        edge_input: np.ndarray | None = None,
        want_cam: bool = False,
    ) -> dict[str, Any]:
        """가능하면 ensemble, 아니면 color 단일.

        Args:
            color_input: (1, 3, 224, 224) — 항상 필요
            edge_input: (1, 3, 224, 224) — has_edge_stream 시에만 사용
            want_cam: True 면 응답에 'cam' (np.ndarray, (H, W)) 포함.
                      color 모델이 cam-aware 일 때만 의미 있음.
        """
        t0 = time.perf_counter()

        color_logits, color_cam_all = self._run_color_with_cam(color_input)
        color_probs = _softmax(color_logits)[0]

        if self.has_edge_stream and edge_input is not None:
            edge_logits = self._run(self.edge_session, self.edge_input_name, edge_input)
            edge_probs = _softmax(edge_logits)[0]
            # Shape sanity — color/edge 가 다른 클래스 수로 학습됐을 수 있음
            # (예: color 만 retrain 했고 edge 는 옛 6-class 그대로).
            # 그 경우 ensemble 불가, color 단독으로 fallback.
            if edge_probs.shape == color_probs.shape:
                # Late fusion — weighted ensemble.
                # color 0.8 / edge 0.2 가 test set 에서 최적 (92.61%, color 단독 91.82%)
                probs = (
                    config.ENSEMBLE_COLOR_WEIGHT * color_probs
                    + (1.0 - config.ENSEMBLE_COLOR_WEIGHT) * edge_probs
                )
                mode = (
                    f"ensemble (color={config.ENSEMBLE_COLOR_WEIGHT:.1f}, "
                    f"edge={1 - config.ENSEMBLE_COLOR_WEIGHT:.1f})"
                )
            else:
                probs = color_probs
                mode = (
                    f"single (color — edge skipped: "
                    f"shape {edge_probs.shape} ≠ {color_probs.shape})"
                )
        else:
            probs = color_probs
            mode = "single (color)"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        idx = int(probs.argmax())
        # 모델 출력 길이와 라벨 수가 일치해야 하지만, 미스매치 시 안전하게 클램프
        labels = self.labels
        n_match = min(len(labels), len(probs))
        result: dict[str, Any] = {
            "predicted_class": labels[idx] if idx < len(labels) else f"class_{idx}",
            "predicted_index": idx,
            "confidence": float(probs[idx]),
            "all_probabilities": {
                labels[i]: float(probs[i])
                for i in range(n_match)
            },
            "model_arch": mode,
            "inference_ms": round(elapsed_ms, 2),
        }
        if want_cam and color_cam_all is not None:
            # color stream 의 cam 만 사용 — ensemble 의 top 클래스에 대해
            # (cam 은 color 만 갖고 있음; edge 는 단일 출력).
            # 안전: 모델 출력 N 채널과 CLASS_LABELS 길이가 어긋날 수 있어 bounds check.
            num_cam_classes = color_cam_all.shape[1]
            if 0 <= idx < num_cam_classes:
                result["cam"] = color_cam_all[0, idx]  # (H, W), np.ndarray
            else:
                # 인덱스 매핑이 어긋난 비정상 상태 — silently skip cam
                log.warning(f"cam idx {idx} 가 모델 출력 채널 수 ({num_cam_classes}) 범위 밖")
        return result


@lazy_singleton
def _load() -> tuple[WasteClassifier, Any]:
    """첫 호출 시 Supabase 의 active 버전을 fetch (있으면) 후 로드.
    active meta 의 class_labels 가 있으면 그것을 사용 (동적 N 클래스 지원)."""
    from src.model_loader import resolve_model_paths
    color_path, edge_path, meta = resolve_model_paths()
    labels = meta.class_labels if meta and meta.class_labels else None
    classifier = WasteClassifier(
        model_path=color_path,
        edge_model_path=edge_path,
        labels=labels,
    )
    return classifier, meta  # meta: RemoteModelMeta | None — None 이면 fallback (config) 사용 중


def get_classifier() -> WasteClassifier:
    return _load()[0]


def get_active_meta():
    """현재 로드된 모델의 RemoteModelMeta (fallback 모드면 None). 로드를 유발하지 않음."""
    return _load.instance[1] if _load.instance else None


def reset_classifier() -> None:
    """캐시된 인스턴스 폐기. 다음 get_classifier() 호출 시 재로드 (model_loader 재실행)."""
    _load.reset()
