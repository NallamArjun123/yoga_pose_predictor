"""
Visualizer — draws OpenPose skeleton, keypoint dots, and pose label overlay
onto BGR frames.
"""

import cv2
import numpy as np
from utils.pose_detector import COCO_POSE_PAIRS, NUM_POINTS


# Colour palette for skeleton limbs (BGR)
LIMB_COLORS = [
    (0,   255, 0),   (0,   255, 85),  (0,   255, 170), (0,   255, 255),
    (0,   170, 255), (0,   85,  255), (0,   0,   255), (85,  0,   255),
    (170, 0,   255), (255, 0,   255), (255, 0,   170), (255, 0,   85),
    (255, 0,   0),   (255, 85,  0),   (255, 170, 0),   (255, 255, 0),
    (170, 255, 0),   (85,  255, 0),   (0,   255, 0),   (0,   255, 0),
    (0,   255, 85),  (0,   170, 255), (0,   85,  255), (0,   0,   255),
]

KEYPOINT_COLOR = (0, 200, 255)   # orange-ish dot for each detected joint

# Pose label colours (BGR) — green for known, amber for Unknown
LABEL_COLOR_MAP = {
    "Unknown": (0, 165, 255),
}
DEFAULT_LABEL_COLOR = (50, 220, 50)


def draw_skeleton(frame: np.ndarray, keypoints: list, confidence: list) -> np.ndarray:
    """
    Draw keypoint dots and skeleton limbs on the frame.

    Args:
        frame      : BGR image (modified in-place and returned)
        keypoints  : list of (x, y) or None — length NUM_POINTS
        confidence : list of float scores

    Returns:
        annotated frame
    """
    # Draw limbs
    for idx, (partA, partB) in enumerate(COCO_POSE_PAIRS):
        if partA >= NUM_POINTS or partB >= NUM_POINTS:
            continue
        kpA = keypoints[partA]
        kpB = keypoints[partB]
        if kpA is None or kpB is None:
            continue
        color = LIMB_COLORS[idx % len(LIMB_COLORS)]
        cv2.line(frame, kpA, kpB, color, 3, cv2.LINE_AA)

    # Draw keypoint dots
    for i, kp in enumerate(keypoints):
        if kp is None:
            continue
        cv2.circle(frame, kp, 5, KEYPOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, kp, 7, (0, 0, 0), 1, cv2.LINE_AA)  # thin outline

    return frame


def draw_pose_label(frame: np.ndarray, pose_name: str, score: float) -> np.ndarray:
    """
    Overlay a semi-transparent banner with the detected pose name and confidence.

    Args:
        frame     : BGR image
        pose_name : classified pose string
        score     : confidence 0–100

    Returns:
        annotated frame
    """
    h, w = frame.shape[:2]
    banner_h = 60
    overlay  = frame.copy()

    # Dark banner at top
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = LABEL_COLOR_MAP.get(pose_name, DEFAULT_LABEL_COLOR)

    # Pose name
    label = f"Pose: {pose_name}"
    cv2.putText(frame, label, (15, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y   = w - 220, 18
    bar_w, bar_h   = 200, 24
    filled          = int(bar_w * min(score, 100) / 100)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    pct_text = f"{score:.1f}%"
    cv2.putText(frame, pct_text, (bar_x + bar_w + 6, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter in bottom-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (w - 120, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return frame
