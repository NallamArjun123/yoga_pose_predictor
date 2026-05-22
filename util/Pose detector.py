"""
PoseDetector — wraps OpenCV's DNN module to run OpenPose (COCO body_25 or MPI model).

Model files required (place in models/ directory):
  COCO body_25 :
    pose_deploy.prototxt
    pose_iter_440000.caffemodel

  MPI (lighter, 15 keypoints):
    pose_deploy_linevec_faster_4_stages.prototxt
    pose_iter_160000.caffemodel

Download from the official OpenPose repo or Hugging Face mirrors.
"""

import cv2
import numpy as np
import os


# ── COCO Body-25 keypoint names & skeleton pairs ──────────────────────────────
COCO_BODY_PARTS = {
    0: "Nose",      1: "Neck",      2: "RShoulder", 3: "RElbow",
    4: "RWrist",    5: "LShoulder", 6: "LElbow",    7: "LWrist",
    8: "MidHip",    9: "RHip",     10: "RKnee",    11: "RAnkle",
   12: "LHip",     13: "LKnee",   14: "LAnkle",   15: "REye",
   16: "LEye",     17: "REar",    18: "LEar",     19: "LBigToe",
   20: "LSmallToe",21: "LHeel",   22: "RBigToe",  23: "RSmallToe",
   24: "RHeel",    25: "Background"
}

COCO_POSE_PAIRS = [
    [1, 2],  [1, 5],  [2, 3],  [3, 4],  [5, 6],  [6, 7],
    [1, 8],  [8, 9],  [9, 10], [10,11], [8,12],  [12,13],
    [13,14], [1,0],   [0,15],  [15,17], [0,16],  [16,18],
    [14,19], [19,20], [14,21], [11,22], [22,23], [11,24],
]

NUM_POINTS = 25


class PoseDetector:
    """
    Loads an OpenPose Caffe model and runs inference on BGR frames.

    Returns:
        keypoints  : list of (x, y) or None for each of NUM_POINTS body parts
        confidence : list of float confidence scores (0.0–1.0)
    """

    def __init__(
        self,
        model_dir: str = "models",
        proto_file: str = "pose_deploy.prototxt",
        weights_file: str = "pose_iter_440000.caffemodel",
        in_width: int = 368,
        in_height: int = 368,
        threshold: float = 0.1,
    ):
        self.threshold = threshold
        self.in_width  = in_width
        self.in_height = in_height

        proto   = os.path.join(model_dir, proto_file)
        weights = os.path.join(model_dir, weights_file)

        if not os.path.isfile(proto) or not os.path.isfile(weights):
            raise FileNotFoundError(
                f"\n[PoseDetector] Model files not found!\n"
                f"  Expected:\n    {proto}\n    {weights}\n\n"
                f"  Download from:\n"
                f"    http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/\n"
                f"  or use the download helper:\n"
                f"    python utils/download_models.py\n"
            )

        self.net = cv2.dnn.readNetFromCaffe(proto, weights)

        # Prefer GPU if available
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[PoseDetector] Using CUDA backend.")
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[PoseDetector] Using CPU backend.")

    def detect(self, frame: np.ndarray):
        """
        Run OpenPose on a single BGR frame.

        Returns:
            keypoints  : list[(x, y) | None]  — length = NUM_POINTS
            confidence : list[float]           — length = NUM_POINTS
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255,
            size=(self.in_width, self.in_height),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()  # shape: (1, 57, H/8, W/8) for COCO

        out_h = output.shape[2]
        out_w = output.shape[3]

        keypoints  = []
        confidence = []

        for i in range(NUM_POINTS):
            prob_map = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(prob_map)

            x = int((point[0] / out_w) * w)
            y = int((point[1] / out_h) * h)

            if conf > self.threshold:
                keypoints.append((x, y))
                confidence.append(float(conf))
            else:
                keypoints.append(None)
                confidence.append(0.0)

        return keypoints, confidence

    @staticmethod
    def get_pose_pairs():
        return COCO_POSE_PAIRS

    @staticmethod
    def get_body_parts():
        return COCO_BODY_PARTS
