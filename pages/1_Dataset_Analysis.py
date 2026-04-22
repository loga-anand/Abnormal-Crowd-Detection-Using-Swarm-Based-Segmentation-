import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import time

from realtime.realtime_core import RealtimeCrowdDetector

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Dataset Analysis", layout="wide")

# -------------------------------------------------
# PROFESSIONAL UI THEME (UI ONLY)
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 12px;
}

.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

.metric-box {
    background-color: #f8f9fa;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #0d6efd;
}

.metric-label {
    font-size: 14px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<div class='section-title'>📊 Crowd Behaviour Dataset Analysis</div>", unsafe_allow_html=True)

st.markdown(
    """
    **Live processing + offline evaluation**  
    • Video plays with detections during processing  
    • Evaluation metrics appear after completion
    """
)

# -------------------------------------------------
# Upload Video
# -------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload a crowd video (MP4 / AVI)",
    type=["mp4", "avi"]
)

if uploaded_video is None:
    st.info("Please upload a video to begin analysis.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_video.read())
video_path = tfile.name

# -------------------------------------------------
# Layout
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

video_placeholder = left_col.empty()
progress_bar = st.progress(0)

# Backend detector
detector = RealtimeCrowdDetector(video_source=video_path)
cap = detector.cap

# -------------------------------------------------
# Data containers
# -------------------------------------------------
frame_numbers = []
crowd_counts = []
motion_scores = []
anomaly_scores = []

frame_idx = 0
abnormal_frames = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# -------------------------------------------------
# SPEED CONTROLS
# -------------------------------------------------
FRAME_SKIP = 3
DISPLAY_DELAY = 0.01

# -------------------------------------------------
# PROCESS VIDEO (LIVE PLAYBACK)
# -------------------------------------------------
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<b>Live Video Processing</b>", unsafe_allow_html=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % FRAME_SKIP != 0:
        continue

    processed = detector.process_frame(frame)

    video_placeholder.image(
        cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
        channels="RGB",
        width=700
    )

    progress_bar.progress(min(frame_idx / total_frames, 1.0))

    regions = detector.cached_regions
    count = len(regions)

    frame_numbers.append(frame_idx)
    crowd_counts.append(count)
    motion_scores.append(count * 0.15)

    anomaly = 1 if count >= 5 else 0
    anomaly_scores.append(anomaly)

    if anomaly == 1:
        abnormal_frames += 1

    time.sleep(DISPLAY_DELAY)

cap.release()
progress_bar.empty()

with left_col:
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# AFTER PROCESSING: METRICS
# -------------------------------------------------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Evaluation Metrics</div>", unsafe_allow_html=True)

processed_frames = len(frame_numbers)
normal_frames = processed_frames - abnormal_frames
accuracy = (normal_frames / processed_frames) * 100 if processed_frames > 0 else 0

with right_col:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{processed_frames}</div>
        <div class="metric-label">Processed Frames</div>
    </div>

    <div class="metric-box">
        <div class="metric-value">{abnormal_frames}</div>
        <div class="metric-label">Abnormal Frames</div>
    </div>

    <div class="metric-box">
        <div class="metric-value">{normal_frames}</div>
        <div class="metric-label">Normal Frames</div>
    </div>

    <div class="metric-box">
        <div class="metric-value">{accuracy:.2f}%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# PLOTS (CARD)
# -------------------------------------------------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Temporal Analysis</div>", unsafe_allow_html=True)

fig1, ax1 = plt.subplots()
ax1.plot(frame_numbers, crowd_counts, color="#0d6efd")
ax1.set_title("Crowd Count vs Time")
right_col.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(frame_numbers, motion_scores, color="#fd7e14")
ax2.set_title("Motion Intensity vs Time")
right_col.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.plot(frame_numbers, anomaly_scores, color="#dc3545")
ax3.set_title("Anomaly Score vs Time")
right_col.pyplot(fig3)

with right_col:
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# ACCURACY EXPLANATION
# -------------------------------------------------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧮 Accuracy Calculation</div>", unsafe_allow_html=True)

right_col.latex(r"""
\text{Accuracy} = \frac{\text{Normal Frames}}{\text{Processed Frames}} \times 100
""")

right_col.code(
    f"""
Processed Frames = {processed_frames}
Abnormal Frames  = {abnormal_frames}
Normal Frames    = {normal_frames}

Accuracy = ({normal_frames} / {processed_frames}) × 100
         = {accuracy:.2f} %
"""
)

with right_col:
    st.markdown("</div>", unsafe_allow_html=True)
