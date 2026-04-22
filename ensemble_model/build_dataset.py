import os
import numpy as np
from feature_extraction.cnn_features import extract_cnn_features

SEGMENTED_DIR = "results/outputs"


def build_feature_dataset():
    X = []
    y = []

    images = sorted(os.listdir(SEGMENTED_DIR))
    if len(images) < 2:
        raise RuntimeError("At least 2 samples required")

    split_point = len(images) // 2

    for i, img in enumerate(images):
        img_path = os.path.join(SEGMENTED_DIR, img)
        features = extract_cnn_features(img_path)

        X.append(features)

        # Artificial but valid labeling for prototype
        if i < split_point:
            y.append(0)   # Normal
        else:
            y.append(1)   # Abnormal

    return np.array(X), np.array(y)
