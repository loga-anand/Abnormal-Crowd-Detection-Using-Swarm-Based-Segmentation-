import cv2
import os

INPUT_DIR = "dataset/frames"
OUTPUT_DIR = "dataset/preprocessed_frames"
IMAGE_SIZE = (224, 224)


def preprocess_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = os.listdir(INPUT_DIR)
    if len(images) == 0:
        raise RuntimeError("No frames found in dataset/frames")

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)

        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), resized)

    print("[INFO] Frame preprocessing completed successfully")


if __name__ == "__main__":
    preprocess_frames()
