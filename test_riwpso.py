import os
import cv2
from swarm_segmentation.riwpso_segmentation import run_riwpso

INPUT_DIR = "dataset/preprocessed_frames"
OUTPUT_DIR = "results/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = os.listdir(INPUT_DIR)
if len(frames) == 0:
    raise RuntimeError("No preprocessed frames found")

for i, frame in enumerate(frames[:20]):  # segment first 20 frames
    frame_path = os.path.join(INPUT_DIR, frame)
    segmented = run_riwpso(frame_path)

    out_path = os.path.join(OUTPUT_DIR, f"segmented_{frame}")
    cv2.imwrite(out_path, segmented)

    print(f"[INFO] Segmented {i+1}/{len(frames[:20])}")

print("[DONE] Batch RIWPSO segmentation completed")
