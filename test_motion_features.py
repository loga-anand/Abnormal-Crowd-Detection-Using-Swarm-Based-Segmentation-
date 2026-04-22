from feature_extraction.motion_features import MotionFeatureExtractor
import cv2

cap = cv2.VideoCapture("dataset/videos/sample.mp4")
motion = MotionFeatureExtractor()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    regions = [(50, 50, 200, 200)]  # dummy region
    feats = motion.extract_features(frame, regions)
    print(feats)

cap.release()
