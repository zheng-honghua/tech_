"""Recover visible RGB silhouettes without fabricating depth observations."""
from __future__ import annotations

import cv2
import numpy as np


def recover_rgb_masks(
    image_bgr: np.ndarray, depth_masks: list[np.ndarray], roi: np.ndarray,
) -> list[np.ndarray]:
    """Associate a saturated RGB component only when it has one depth owner.

    Ambiguous touching objects and achromatic objects retain their depth mask.
    Returned masks are for RGB display/colour only, never for suction planning.
    """
    if not depth_masks:
        return []
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    candidate = ((hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 25) & (roi > 0)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidate[roi == 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    output = [mask.copy() for mask in depth_masks]
    for label in range(1, count):
        region = labels == label
        owners = [i for i, mask in enumerate(depth_masks)
                  if np.count_nonzero(region & (mask > 0)) >= max(20, np.count_nonzero(mask) * 0.1)]
        if len(owners) != 1:
            continue
        index = owners[0]
        area = np.count_nonzero(depth_masks[index])
        overlap = np.count_nonzero(region & (depth_masks[index] > 0))
        if overlap < area * 0.65 or stats[label, cv2.CC_STAT_AREA] > area * 3:
            continue
        # Keep all source depth pixels and add the observed RGB surface only.
        output[index][region] = 255
        output[index][roi == 0] = 0
    return output
