from preprocessing.video_to_frames import extract_frames_from_all_videos
from preprocessing.frame_preprocessing import preprocess_frames
from swarm_segmentation.riwpso_segmentation import run_riwpso
from ensemble_model.build_dataset import build_feature_dataset
from ensemble_model.ensemble_classifier import train_ensemble
from evaluation.metrics import evaluate_model

import os
import cv2

print("\n===== FINAL YEAR PROJECT PIPELINE STARTED =====\n")

# STEP 1: Video → Frames
print("[STEP 1] Extracting frames from videos")
extract_frames_from_all_videos()

# STEP 2: Frame preprocessing
print("[STEP 2] Preprocessing frames")
preprocess_frames()

# STEP 3: RIWPSO segmentation (batch)
print("[STEP 3] RIWPSO-based segmentation")
input_dir = "dataset/preprocessed_frames"
output_dir = "results/outputs"
os.makedirs(output_dir, exist_ok=True)

for img in os.listdir(input_dir)[:20]:
    img_path = os.path.join(input_dir, img)
    segmented = run_riwpso(img_path)
    cv2.imwrite(os.path.join(output_dir, f"segmented_{img}"), segmented)

# STEP 4 & 5: Feature extraction + Ensemble learning
print("[STEP 4 & 5] Feature extraction and ensemble learning")
X, y = build_feature_dataset()
model, y_test, y_pred, y_prob = train_ensemble(X, y)

# STEP 6: Evaluation
print("[STEP 6] Model evaluation")
evaluate_model(y_test, y_pred, y_prob)

print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")
