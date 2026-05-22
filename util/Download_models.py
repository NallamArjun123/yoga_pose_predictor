"""
Download OpenPose COCO model files into the models/ directory.

Files downloaded:
  • pose_deploy.prototxt         (~  28 KB)
  • pose_iter_440000.caffemodel  (~ 200 MB)

Run once before using the detector:
    python utils/download_models.py
"""

import os
import urllib.request


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FILES = {
    "pose_deploy.prototxt": (
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/"
        "master/models/pose/coco/pose_deploy_linevec.prototxt"
    ),
    "pose_iter_440000.caffemodel": (
        "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/"
        "pose_iter_440000.caffemodel"
    ),
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = downloaded * 100 / total_size if total_size > 0 else 0
    bar = int(pct / 2)
    print(f"\r  [{'#' * bar}{' ' * (50 - bar)}] {pct:5.1f}%", end="", flush=True)


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading OpenPose COCO model files ...\n")

    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.isfile(dest):
            print(f"  ✓ {filename} already exists, skipping.")
            continue
        print(f"  ↓ {filename}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print(f"\n  ✓ Saved → {dest}")
        except Exception as e:
            print(f"\n  ✗ Failed: {e}")
            print(f"    Please download manually from:\n    {url}")

    print("\nDone.")


if __name__ == "__main__":
    download_models()
