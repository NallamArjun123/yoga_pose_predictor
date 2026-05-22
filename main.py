"""
Yoga Pose Detection using OpenPose + OpenCV
Supports: Webcam (real-time), Video file, Image file
"""

import cv2
import argparse
import os
import sys
from utils.pose_detector import PoseDetector
from utils.yoga_classifier import YogaClassifier
from utils.visualizer import draw_skeleton, draw_pose_label


def process_image(image_path: str, detector: PoseDetector, classifier: YogaClassifier, output_dir: str):
    """Process a single image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    keypoints, confidence = detector.detect(frame)
    pose_name, score = classifier.classify(keypoints)

    output = draw_skeleton(frame.copy(), keypoints, confidence)
    output = draw_pose_label(output, pose_name, score)

    out_path = os.path.join(output_dir, "result_" + os.path.basename(image_path))
    cv2.imwrite(out_path, output)
    print(f"[IMAGE] Detected pose: {pose_name} ({score:.1f}%) → saved to {out_path}")

    cv2.imshow("Yoga Pose Detection - Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path: str, detector: PoseDetector, classifier: YogaClassifier, output_dir: str):
    """Process a video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = os.path.join(output_dir, "result_" + os.path.basename(video_path))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[VIDEO] Processing {video_path} ...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, confidence = detector.detect(frame)
        pose_name, score = classifier.classify(keypoints)

        output = draw_skeleton(frame.copy(), keypoints, confidence)
        output = draw_pose_label(output, pose_name, score)
        writer.write(output)

        cv2.imshow("Yoga Pose Detection - Video", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[VIDEO] Interrupted by user.")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: {pose_name} ({score:.1f}%)")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[VIDEO] Done. Output saved to {out_path}")


def process_webcam(detector: PoseDetector, classifier: YogaClassifier, cam_index: int = 0):
    """Real-time yoga pose detection from webcam."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[WEBCAM] Press 'q' to quit, 's' to save a snapshot.")

    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        keypoints, confidence = detector.detect(frame)
        pose_name, score = classifier.classify(keypoints)

        output = draw_skeleton(frame.copy(), keypoints, confidence)
        output = draw_pose_label(output, pose_name, score)

        cv2.imshow("Yoga Pose Detection - Webcam", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[WEBCAM] Exiting.")
            break
        elif key == ord("s"):
            snap_path = os.path.join("output", f"snapshot_{snapshot_count:04d}.jpg")
            cv2.imwrite(snap_path, output)
            print(f"[WEBCAM] Snapshot saved → {snap_path}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Yoga Pose Detection using OpenPose + OpenCV"
    )
    parser.add_argument(
        "--mode",
        choices=["webcam", "video", "image"],
        default="webcam",
        help="Input mode: webcam | video | image (default: webcam)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video or image file (required for video/image mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for saved results (default: output/)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing OpenPose model files (default: models/)"
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Keypoint confidence threshold (default: 0.1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 55)
    print("  Yoga Pose Detection — OpenPose + OpenCV")
    print("=" * 55)
    print(f"  Mode       : {args.mode}")
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Output dir : {args.output}")
    print(f"  Threshold  : {args.threshold}")
    print("=" * 55)

    # Initialize detector and classifier
    detector   = PoseDetector(model_dir=args.model_dir, threshold=args.threshold)
    classifier = YogaClassifier()

    if args.mode == "webcam":
        process_webcam(detector, classifier, cam_index=args.cam)

    elif args.mode == "image":
        if not args.input:
            print("[ERROR] --input is required for image mode.")
            sys.exit(1)
        process_image(args.input, detector, classifier, args.output)

    elif args.mode == "video":
        if not args.input:
            print("[ERROR] --input is required for video mode.")
            sys.exit(1)
        process_video(args.input, detector, classifier, args.output)


if __name__ == "__main__":
    main()
