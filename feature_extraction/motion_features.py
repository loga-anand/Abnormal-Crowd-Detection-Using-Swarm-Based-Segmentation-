import cv2
import numpy as np

class MotionFeatureExtractor:
    def __init__(self):
        self.prev_gray = None

    def compute_optical_flow(self, prev_gray, curr_gray):
        return cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

    def extract_features(self, frame, regions):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return []

        flow = self.compute_optical_flow(self.prev_gray, gray)
        self.prev_gray = gray

        features = []

        for (x, y, w, h) in regions:
            roi_flow = flow[y:y+h, x:x+w]
            if roi_flow.size == 0:
                continue

            mag, ang = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])

            feat = {
                "motion_mag_mean": float(np.mean(mag)),
                "motion_mag_std": float(np.std(mag)),
                "motion_dir_std": float(np.std(ang)),
                "motion_energy": float(np.sum(mag)),
                "region_area": int(w * h)
            }
            features.append(feat)

        return features
