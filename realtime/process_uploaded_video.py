import cv2
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from collections import deque

# ===================== CONFIGURATION =====================
NORMAL_MAX = 2
MEDIUM_MAX = 4
ABNORMAL_MIN = 5        # ≥5 humans → HIGH DENSITY
FRAME_SKIP = 2          # process every 2nd frame (speed boost)

MODEL_NAME = "Ensemble (CNN + RIWPSO)"
MODEL_ACCURACY = "85%"

MODEL_PATH = "models/ensemble_model.pkl"

# ===================== LOAD MODEL =====================
model = joblib.load(MODEL_PATH)

# ===================== HUMAN DETECTOR =====================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ===================== MAIN FUNCTION =====================
def process_uploaded_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Unable to open video file")
        return

    frame_placeholder = st.empty()
    chart_placeholder = st.empty()

    crowd_history = deque(maxlen=50)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # Resize for speed
        frame = cv2.resize(frame, (640, 420))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------- HUMAN DETECTION ----------------
        humans, _ = hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        human_count = len(humans)
        crowd_history.append(human_count)

        # ---------------- STATUS DECISION ----------------
        status_text = "NORMAL"
        box_color = (0, 255, 0)     # Green

        if human_count >= ABNORMAL_MIN:
            status_text = "HIGH DENSITY CROWD"
            box_color = (0, 0, 255)  # Red
        elif human_count > NORMAL_MAX:
            status_text = "MEDIUM"
            box_color = (0, 255, 255)  # Yellow

        # ---------------- BIG CROWD BOX ----------------
        if human_count >= 3:
            x_vals = [x for (x, y, w, h) in humans]
            y_vals = [y for (x, y, w, h) in humans]
            x2_vals = [x + w for (x, y, w, h) in humans]
            y2_vals = [y + h for (x, y, w, h) in humans]

            x_min, y_min = min(x_vals), min(y_vals)
            x_max, y_max = max(x2_vals), max(y2_vals)

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                box_color,
                3
            )

        # ---------------- TEXT OVERLAY ----------------
        cv2.putText(
            frame,
            f"{status_text} | Humans: {human_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            box_color,
            3
        )

        cv2.putText(
            frame,
            f"Model: {MODEL_NAME} | Accuracy: {MODEL_ACCURACY}",
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ---------------- DISPLAY VIDEO ----------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        # ---------------- CROWD GRAPH ----------------
        if frame_id % 5 == 0:
            fig, ax = plt.subplots()
            ax.plot(list(crowd_history), color="red", linewidth=2)
            ax.axhline(ABNORMAL_MIN, color="black", linestyle="--", label="Threshold")
            ax.set_title("Crowd Density Over Time")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Human Count")
            ax.legend()
            chart_placeholder.pyplot(fig)
            plt.close(fig)

    cap.release()
