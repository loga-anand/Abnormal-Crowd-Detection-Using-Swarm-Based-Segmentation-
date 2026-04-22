import cv2
import numpy as np
import joblib
from collections import deque
from feature_extraction.cnn_features import extract_cnn_features
from scipy.spatial.distance import cdist

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = joblib.load("models/ensemble_model.pkl")

# ===============================
# HUMAN DETECTOR (HOG + SVM)
# ===============================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ===============================
# CAMERA SETUP
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Multi-crowd human detection started. Press Q to quit.")

# ===============================
# TEMPORAL BUFFERS (ONE PER CROWD)
# ===============================
crowd_buffers = {}

# ===============================
# HELPER: GROUP HUMANS INTO CROWDS
# ===============================
def group_humans(rects, distance_thresh=120):
    centers = np.array([
        (x + w // 2, y + h // 2) for (x, y, w, h) in rects
    ])

    if len(centers) == 0:
        return []

    dist_matrix = cdist(centers, centers)
    groups = []
    visited = set()

    for i in range(len(centers)):
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        for j in range(len(centers)):
            if j not in visited and dist_matrix[i][j] < distance_thresh:
                group.append(j)
                visited.add(j)

        if len(group) >= 2:  # at least 2 humans = crowd
            groups.append(group)

    return groups

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()

    # -------------------------------
    # STEP 1: HUMAN DETECTION
    # -------------------------------
    rects, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    if len(rects) < 2:
        cv2.putText(
            output,
            "NO HUMAN CROWD",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )
        cv2.imshow("Multi-Crowd Human Monitoring", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # -------------------------------
    # STEP 2: GROUP HUMANS → CROWDS
    # -------------------------------
    groups = group_humans(rects)

    if len(groups) == 0:
        cv2.imshow("Multi-Crowd Human Monitoring", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # -------------------------------
    # STEP 3: PROCESS EACH CROWD
    # -------------------------------
    for idx, group in enumerate(groups):
        xs = [rects[i][0] for i in group]
        ys = [rects[i][1] for i in group]
        xe = [rects[i][0] + rects[i][2] for i in group]
        ye = [rects[i][1] + rects[i][3] for i in group]

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xe), max(ye)

        crowd_crop = frame[y1:y2, x1:x2]
        if crowd_crop.size == 0:
            continue

        crowd_crop = cv2.resize(crowd_crop, (224, 224))
        cv2.imwrite("temp_frame.jpg", crowd_crop)

        features = extract_cnn_features("temp_frame.jpg").reshape(1, -1)
        prediction = model.predict(features)[0]

        # -------------------------------
        # TEMPORAL SMOOTHING PER CROWD
        # -------------------------------
        if idx not in crowd_buffers:
            crowd_buffers[idx] = deque(maxlen=10)

        crowd_buffers[idx].append(prediction)
        avg_pred = sum(crowd_buffers[idx]) / len(crowd_buffers[idx])

        if avg_pred > 0.6:
            label = "ABNORMAL HUMAN CROWD"
            color = (0, 0, 255)
        else:
            label = "NORMAL HUMAN CROWD"
            color = (0, 255, 0)

        # -------------------------------
        # DRAW CROWD BOX + LABEL
        # -------------------------------
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            output,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # -------------------------------
    # OPTIONAL: DRAW INDIVIDUAL HUMANS
    # -------------------------------
    for (x, y, w, h) in rects:
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 1)

    cv2.imshow("Multi-Crowd Human Monitoring", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program exited cleanly.")
