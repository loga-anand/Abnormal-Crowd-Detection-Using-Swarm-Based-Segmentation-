import os
from feature_extraction.cnn_features import extract_cnn_features

SEGMENTED_DIR = "results/outputs"

images = os.listdir(SEGMENTED_DIR)
if len(images) == 0:
    raise RuntimeError("No segmented images found")

image_path = os.path.join(SEGMENTED_DIR, images[0])

features = extract_cnn_features(image_path)

print("[INFO] CNN feature extraction successful")
print("[INFO] Feature vector length:", len(features))
