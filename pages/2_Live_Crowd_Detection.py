import streamlit as st
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from realtime.realtime_core import RealtimeCrowdDetector

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Live Crowd Detection", layout="wide")

# -------------------------------------------------
# PROFESSIONAL UI THEME
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

.metric-label {
    font-size: 14px;
    color: #555;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
}

.blue { color: #0d6efd; }
.red { color: #dc3545; }
.green { color: #198754; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<div class='section-title'>🎥 Live Crowd Anomaly Detection</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Layout
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

# -------------------------------------------------
# START / STOP BUTTONS (ABOVE CAMERA)
# -------------------------------------------------
btn1, btn2 = left_col.columns(2)
start = btn1.button("▶️ Start Camera")
stop = btn2.button("⏹️ Stop Camera")

if "run_live" not in st.session_state:
    st.session_state.run_live = False

if start:
    st.session_state.run_live = True
if stop:
    st.session_state.run_live = False

# -------------------------------------------------
# LEFT SIDE PLACEHOLDERS
# -------------------------------------------------
video_placeholder = left_col.empty()
pulse_placeholder = left_col.empty()

# -------------------------------------------------
# RIGHT SIDE DASHBOARD (STATIC STRUCTURE)
# -------------------------------------------------
with right_col:
    st.markdown("<div class='card'><b>Live Metrics</b>", unsafe_allow_html=True)

    crowd_ph = st.empty()
    acc_ph = st.empty()
    total_ph = st.empty()
    abnormal_ph = st.empty()
    status_ph = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Counters
# -------------------------------------------------
total_frames = 0
abnormal_frames = 0

pulse_x = deque(maxlen=60)
pulse_y = deque(maxlen=60)

# -------------------------------------------------
# Run Live Detection
# -------------------------------------------------
if st.session_state.run_live:
    detector = RealtimeCrowdDetector(video_source=0)
    cap = detector.cap

    FRAME_SKIP = 2
    DISPLAY_DELAY = 0.005
    frame_idx = 0

    while cap.isOpened() and st.session_state.run_live:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        processed = detector.process_frame(frame)

        # ---------------- VIDEO ----------------
        video_placeholder.image(
            cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
            channels="RGB",
            width=700
        )

        regions = detector.cached_regions
        crowd_count = len(regions)

        total_frames += 1
        is_abnormal = crowd_count >= 5
        if is_abnormal:
            abnormal_frames += 1

        normal_frames = total_frames - abnormal_frames
        accuracy = (normal_frames / total_frames) * 100 if total_frames else 0

        # ---------------- PULSE ----------------
        pulse_x.append(total_frames)
        pulse_y.append(crowd_count * 0.15)

        fig, ax = plt.subplots()
        ax.plot(pulse_x, pulse_y, color="#dc3545", linewidth=2)
        ax.set_title("Live Crowd Pulse")
        ax.set_ylim(0, max(5, max(pulse_y) + 1))
        pulse_placeholder.pyplot(fig)
        plt.close(fig)

        # ---------------- METRIC UPDATES (NUMBERS ONLY) ----------------
        crowd_ph.markdown(
            f"<div class='metric-label'>Crowd Count</div>"
            f"<div class='metric-value blue'>{crowd_count}</div>",
            unsafe_allow_html=True
        )

        acc_ph.markdown(
            f"<div class='metric-label'>Accuracy</div>"
            f"<div class='metric-value green'>{accuracy:.2f}%</div>",
            unsafe_allow_html=True
        )

        total_ph.markdown(
            f"<div class='metric-label'>Total Frames</div>"
            f"<div class='metric-value blue'>{total_frames}</div>",
            unsafe_allow_html=True
        )

        abnormal_ph.markdown(
            f"<div class='metric-label'>Abnormal Frames</div>"
            f"<div class='metric-value red'>{abnormal_frames}</div>",
            unsafe_allow_html=True
        )

        status_ph.markdown(
            f"<div class='metric-label'>Status</div>"
            f"<div class='metric-value {'red' if is_abnormal else 'green'}'>"
            f"{'ABNORMAL' if is_abnormal else 'NORMAL'}</div>",
            unsafe_allow_html=True
        )

        time.sleep(DISPLAY_DELAY)

    cap.release()
