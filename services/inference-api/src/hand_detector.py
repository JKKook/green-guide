"""MediaPipe Hands 기반 손 영역 감지.

용도: /predict-centered, /predict-with-regions 에서 분류 전 손 영역 제외.
- 손이 이미지 대부분 차지 → non_object 응답 (재촬영 안내)
- 손이 부분 → 손 영역만 mask 처리 후 분류 (객체만 보이게)

원래 의도: 모델이 손을 의류·종이상자 로 잘못 분류하는 문제를 학습 데이터 늘리기 대신
사전학습된 외부 detector 로 surgically 해결.
"""
from __future__ import annotations

import io
import threading

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from src.core.singleton import lazy_singleton


_DILATE_KERNEL = np.ones((15, 15), np.uint8)


class HandDetector:
    """MediaPipe Hands wrapper — image → 손 binary mask."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            # 클로즈업 손바닥/손가락에서 검출률 ↑ 위해 임계 낮춤 (default 0.5).
            min_detection_confidence=0.30,
            model_complexity=1,
        )

    def detect_mask(self, image_bytes: bytes) -> np.ndarray | None:
        """이미지 → 손 mask (H, W) uint8 0/255. 손 없으면 None.

        MediaPipe Hands 만 사용 — 손 전체 구조 (손목·5손가락·손바닥) 가 보일 때 정확.
        Skin color fallback 은 cardboard/paper 같은 베이지 객체에 false positive 가
        커서 채택 안 함. 클로즈업 손 (손바닥 단독, 주먹) 은 검출 안 되지만 객체 분류
        false positive 0 을 우선 (cardboard·paper 가 손으로 잘못 분류되면 더 큰 손해).
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
        rgb = np.array(img)
        H, W = rgb.shape[:2]

        with self._lock:  # MediaPipe instance 가 thread-safe 아님
            result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        mask = np.zeros((H, W), dtype=np.uint8)
        for landmarks in result.multi_hand_landmarks:
            points = np.array([
                [int(lm.x * W), int(lm.y * H)] for lm in landmarks.landmark
            ], dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)
        # 손목·손등 두께 추가 (keypoint 만으로는 너무 좁음)
        mask = cv2.dilate(mask, _DILATE_KERNEL, iterations=1)
        return mask

    def mask_grid(self, image_bytes: bytes, grid: int) -> np.ndarray:
        """grid×grid 손 점유율 (0~1). 손 없으면 0 grid 반환.

        u2netp.object_mask_grid 와 동일 형식 — regions API 에서 결합 가능.
        """
        full = self.detect_mask(image_bytes)
        if full is None:
            return np.zeros((grid, grid), dtype=np.float32)
        small = Image.fromarray(full).resize((grid, grid), Image.BILINEAR)
        return np.array(small).astype(np.float32) / 255.0

    def hand_area_ratio(self, image_bytes: bytes) -> float:
        """손 영역이 전체 이미지의 어느 비율인지 (0~1). 손 없으면 0."""
        mask = self.detect_mask(image_bytes)
        if mask is None:
            return 0.0
        return float((mask > 127).mean())


@lazy_singleton
def get_hand_detector() -> HandDetector:
    return HandDetector()
