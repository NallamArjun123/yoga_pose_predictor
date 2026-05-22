"""
YogaClassifier — rule-based yoga pose classifier using joint angles derived
from OpenPose COCO body-25 keypoints.

Supported poses:
  • Warrior I (Virabhadrasana I)
  • Warrior II (Virabhadrasana II)
  • Tree Pose (Vrksasana)
  • Downward Dog (Adho Mukha Svanasana)
  • Mountain Pose (Tadasana)
  • Cobra Pose (Bhujangasana)
  • Child's Pose (Balasana)
  • Triangle Pose (Trikonasana)
  • Unknown

Each pose is described by a set of angle constraints on specific joints.
The classifier scores each pose and returns the best match.
"""

import math
import numpy as np
from typing import List, Optional, Tuple


# ── Keypoint indices (COCO body-25) ──────────────────────────────────────────
NOSE        = 0
NECK        = 1
R_SHOULDER  = 2
R_ELBOW     = 3
R_WRIST     = 4
L_SHOULDER  = 5
L_ELBOW     = 6
L_WRIST     = 7
MID_HIP     = 8
R_HIP       = 9
R_KNEE      = 10
R_ANKLE     = 11
L_HIP       = 12
L_KNEE      = 13
L_ANKLE     = 14


# ── Helpers ───────────────────────────────────────────────────────────────────

def _angle(a, b, c) -> Optional[float]:
    """Angle (degrees) at vertex b, formed by rays b→a and b→c."""
    if a is None or b is None or c is None:
        return None
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot   = ax * cx + ay * cy
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a * mag_c == 0:
        return None
    cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_val))


def _dist(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _in_range(value, low, high) -> Tuple[bool, float]:
    """Returns (within_range, partial_score 0–1)."""
    if value is None:
        return False, 0.0
    mid    = (low + high) / 2
    half   = (high - low) / 2
    score  = max(0.0, 1.0 - abs(value - mid) / (half + 1e-9))
    return low <= value <= high, score


# ── Pose definitions ──────────────────────────────────────────────────────────

class _PoseRule:
    def __init__(self, name: str):
        self.name = name

    def score(self, kp: list) -> float:
        """Return confidence score 0–100."""
        raise NotImplementedError


class _WarriorI(_PoseRule):
    def __init__(self): super().__init__("Warrior I")

    def score(self, kp):
        scores = []
        # Front knee ~90°
        front_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s = _in_range(front_knee, 75, 110)
        scores.append(s)
        # Back leg mostly straight >150°
        back_knee = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        _, s = _in_range(back_knee, 150, 180)
        scores.append(s)
        # Arms raised — wrists above shoulders
        if kp[L_WRIST] and kp[L_SHOULDER]:
            _, s = _in_range(kp[L_SHOULDER][1] - kp[L_WRIST][1], 20, 300)
            scores.append(s)
        if kp[R_WRIST] and kp[R_SHOULDER]:
            _, s = _in_range(kp[R_SHOULDER][1] - kp[R_WRIST][1], 20, 300)
            scores.append(s)
        return _avg(scores) * 100


class _WarriorII(_PoseRule):
    def __init__(self): super().__init__("Warrior II")

    def score(self, kp):
        scores = []
        # Front knee ~90°
        front_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s = _in_range(front_knee, 75, 110)
        scores.append(s)
        # Arms extended horizontally — elbows near shoulder height
        r_arm = _angle(kp[R_SHOULDER], kp[R_ELBOW], kp[R_WRIST])
        l_arm = _angle(kp[L_SHOULDER], kp[L_ELBOW], kp[L_WRIST])
        _, s1 = _in_range(r_arm, 160, 180)
        _, s2 = _in_range(l_arm, 160, 180)
        scores += [s1, s2]
        return _avg(scores) * 100


class _TreePose(_PoseRule):
    def __init__(self): super().__init__("Tree Pose")

    def score(self, kp):
        scores = []
        # Standing leg straight
        stand = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        _, s = _in_range(stand, 160, 180)
        scores.append(s)
        # Raised leg knee bent out – hip angle
        raised = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s = _in_range(raised, 30, 90)
        scores.append(s)
        # Arms raised
        if kp[L_WRIST] and kp[L_SHOULDER]:
            _, s = _in_range(kp[L_SHOULDER][1] - kp[L_WRIST][1], 30, 400)
            scores.append(s)
        return _avg(scores) * 100


class _DownwardDog(_PoseRule):
    def __init__(self): super().__init__("Downward Dog")

    def score(self, kp):
        scores = []
        # Hip angle (inverted V) — large angle at hip
        hip_ang = _angle(kp[NECK], kp[MID_HIP], kp[R_ANKLE])
        _, s = _in_range(hip_ang, 50, 100)
        scores.append(s)
        # Knees relatively straight
        r_knee = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        l_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s1 = _in_range(r_knee, 150, 180)
        _, s2 = _in_range(l_knee, 150, 180)
        scores += [s1, s2]
        # Hips above shoulder level
        if kp[MID_HIP] and kp[NECK]:
            _, s = _in_range(kp[NECK][1] - kp[MID_HIP][1], 10, 400)
            scores.append(s)
        return _avg(scores) * 100


class _MountainPose(_PoseRule):
    def __init__(self): super().__init__("Mountain Pose (Tadasana)")

    def score(self, kp):
        scores = []
        # Both knees straight
        r_knee = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        l_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s1 = _in_range(r_knee, 160, 180)
        _, s2 = _in_range(l_knee, 160, 180)
        scores += [s1, s2]
        # Arms by sides — elbow angle ~160–180
        r_elbow = _angle(kp[R_SHOULDER], kp[R_ELBOW], kp[R_WRIST])
        l_elbow = _angle(kp[L_SHOULDER], kp[L_ELBOW], kp[L_WRIST])
        _, s3 = _in_range(r_elbow, 155, 180)
        _, s4 = _in_range(l_elbow, 155, 180)
        scores += [s3, s4]
        return _avg(scores) * 100


class _CobraPose(_PoseRule):
    def __init__(self): super().__init__("Cobra Pose")

    def score(self, kp):
        scores = []
        # Spine arched — neck above hips
        if kp[NECK] and kp[MID_HIP]:
            _, s = _in_range(kp[MID_HIP][1] - kp[NECK][1], 30, 400)
            scores.append(s)
        # Elbows bent
        r_elbow = _angle(kp[R_SHOULDER], kp[R_ELBOW], kp[R_WRIST])
        l_elbow = _angle(kp[L_SHOULDER], kp[L_ELBOW], kp[L_WRIST])
        _, s1 = _in_range(r_elbow, 60, 130)
        _, s2 = _in_range(l_elbow, 60, 130)
        scores += [s1, s2]
        # Legs straight on ground
        r_knee = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        l_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s3 = _in_range(r_knee, 155, 180)
        _, s4 = _in_range(l_knee, 155, 180)
        scores += [s3, s4]
        return _avg(scores) * 100


class _ChildsPose(_PoseRule):
    def __init__(self): super().__init__("Child's Pose")

    def score(self, kp):
        scores = []
        # Knees deeply bent
        r_knee = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        l_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s1 = _in_range(r_knee, 20, 70)
        _, s2 = _in_range(l_knee, 20, 70)
        scores += [s1, s2]
        # Head below hips — head y > hip y (image coords)
        if kp[NOSE] and kp[MID_HIP]:
            _, s = _in_range(kp[NOSE][1] - kp[MID_HIP][1], 10, 400)
            scores.append(s)
        return _avg(scores) * 100


class _TrianglePose(_PoseRule):
    def __init__(self): super().__init__("Triangle Pose")

    def score(self, kp):
        scores = []
        # Both knees straight
        r_knee = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        l_knee = _angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        _, s1 = _in_range(r_knee, 155, 180)
        _, s2 = _in_range(l_knee, 155, 180)
        scores += [s1, s2]
        # Side bend — one wrist near ankle
        if kp[R_WRIST] and kp[R_ANKLE]:
            d = _dist(kp[R_WRIST], kp[R_ANKLE])
            ref = _dist(kp[R_HIP], kp[R_ANKLE]) or 1
            _, s = _in_range(1 - d / ref, 0.5, 1.0)
            scores.append(s)
        # Other arm raised
        if kp[L_WRIST] and kp[L_SHOULDER]:
            _, s = _in_range(kp[L_SHOULDER][1] - kp[L_WRIST][1], 20, 400)
            scores.append(s)
        return _avg(scores) * 100


def _avg(lst):
    lst = [x for x in lst if x is not None]
    return sum(lst) / len(lst) if lst else 0.0


# ── Main classifier ───────────────────────────────────────────────────────────

POSE_RULES = [
    _WarriorI(),
    _WarriorII(),
    _TreePose(),
    _DownwardDog(),
    _MountainPose(),
    _CobraPose(),
    _ChildsPose(),
    _TrianglePose(),
]

MIN_SCORE_THRESHOLD = 35.0   # below this → "Unknown"


class YogaClassifier:
    """
    Classify a list of COCO keypoints into a yoga pose name.

    Usage:
        classifier = YogaClassifier()
        pose_name, confidence = classifier.classify(keypoints)
    """

    def classify(self, keypoints: list) -> Tuple[str, float]:
        best_name  = "Unknown"
        best_score = 0.0

        for rule in POSE_RULES:
            try:
                s = rule.score(keypoints)
            except Exception:
                s = 0.0
            if s > best_score:
                best_score = s
                best_name  = rule.name

        if best_score < MIN_SCORE_THRESHOLD:
            return "Unknown", best_score

        return best_name, best_score

    @staticmethod
    def pose_names() -> List[str]:
        return [r.name for r in POSE_RULES] + ["Unknown"]
