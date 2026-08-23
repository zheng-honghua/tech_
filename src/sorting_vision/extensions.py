from __future__ import annotations

from typing import Any, Callable, Protocol

import cv2
import numpy as np


class VisionExtension(Protocol):
    name: str

    def analyze(self, image: np.ndarray, mask: np.ndarray) -> dict[str, Any]: ...


class QRCodeExtension:
    name = "qrcode"

    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def analyze(self, image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        text, points, _ = self.detector.detectAndDecode(image)
        return {
            "text": text,
            "detected": bool(text or points is not None),
            "confidence": 1.0 if text else 0.0,
        }


class CallableExtension:
    """Adapter for OCR or defect models without coupling their dependencies."""

    def __init__(
        self,
        name: str,
        analyzer: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    ) -> None:
        self.name = name
        self.analyzer = analyzer

    def analyze(self, image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        return self.analyzer(image, mask)

