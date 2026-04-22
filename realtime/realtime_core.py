import cv2
import numpy as np
import os
import tempfile
import joblib
from collections import deque

# -------------------------------------------------
# RIWPSO (FUNCTION-BASED)
# -------------------------------------------------
from swarm_segmentation.riwpso_segmentation import run_riwpso

# -------------------------------------------------
# CNN FEATURES (FUNCTION-BASED MobileNetV2)
# -------------------------------------------------
from feature_extraction.cnn_features import extract_cnn_features

# -------------------------------------------------
# OTHER MODULES
# -------------------------------------------------
from feature_extraction.motion_features import MotionFeatureExtractor
from ensemble_model.ensemble_voter import EnsembleVoter


class RealtimeCrowdDetector:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)

        # Feature modules
        self.motion_extractor = MotionFeatureExtractor()
        self.ensemble_model = self.load_ensemble_model()
        self.voter = EnsembleVoter()

        # -----------------------------
        # OPTIMIZATION CONTROLS
        # -----------------------------
        self.frame_count = 0
        self.riwpso_interval = 20      # 🔥 run RIWPSO every 20 frames
        self.cached_regions = []       # reuse last segmentation

    # -------------------------------------------------
    # LOAD TRAINED ENSEMBLE MODEL
    # -------------------------------------------------
    def load_ensemble_model(self):
        model_path = "models/ensemble_model.pkl"
        if not os.path.exists(model_path):
            print("[WARNING] Ensemble model not found. Rule-based only.")
            return None

        print("[INFO] Loading ensemble model (CNN-only)...")
        return joblib.load(model_path)

    # -------------------------------------------------
    # RIWPSO SEGMENTATION (CACHED)
    # -------------------------------------------------
    def riwpso_segment_frame(self, frame):
        """
        Runs RIWPSO only every N frames.
        Reuses cached result in between.
        """
        if self.frame_count % self.riwpso_interval != 0 and self.cached_regions:
            return self.cached_regions

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, gray)
            segmented = run_riwpso(tmp.name)

        os.remove(tmp.name)

        contours, _ = cv2.findContours(
            segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 800:
                regions.append((x, y, w, h))

        self.cached_regions = regions
        return regions

    # -------------------------------------------------
    # PROCESS FRAME
    # -------------------------------------------------
    def process_frame(self, frame):
        self.frame_count += 1

        # 1. RIWPSO segmentation (optimized)
        regions = self.riwpso_segment_frame(frame)

        # 2. CNN feature extraction (CNN-only for ML)
        cnn_features = []
        for (x, y, w, h) in regions:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            cnn_features.append(extract_cnn_features(roi))

        # 3. Motion features (rule-level)
        motion_features = []
        if regions:
            motion_features = self.motion_extractor.extract_features(frame, regions)

        # 4. Ensemble ML score (CNN-only)
        ml_score = 0.0
        if self.ensemble_model is not None and len(cnn_features) > 0:
            ml_score = self.ensemble_model.predict_proba(
                np.array(cnn_features)
            )[0][1]

        # 5. Rule-based score
        rule_score = 0.0
        if len(regions) >= 5:
            rule_score += 0.6
        if motion_features:
            rule_score += 0.4

        # 6. Final decision
        final_decision = self.voter.vote(
            rule_score=rule_score,
            svm_score=ml_score,
            rf_score=ml_score,
            lstm_score=0.0
        )

        label = "ABNORMAL" if final_decision == 1 else "NORMAL"

        # 7. Visualization
        for (x, y, w, h) in regions:
            color = (0, 0, 255) if label == "ABNORMAL" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(
            frame,
            f"STATUS: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if label == "ABNORMAL" else (0, 255, 0),
            2
        )

        return frame

    # -------------------------------------------------
    # REALTIME LOOP
    # -------------------------------------------------
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            output = self.process_frame(frame)
            cv2.imshow(
                "Crowd Anomaly Detection (Optimized RIWPSO)",
                output
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = RealtimeCrowdDetector(video_source=0)
    detector.run()
