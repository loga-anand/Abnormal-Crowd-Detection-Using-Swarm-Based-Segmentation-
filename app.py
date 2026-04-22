import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Crowd Anomaly Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# BACKGROUND (DARKER FOR VISIBILITY)
# -------------------------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-image: url("data:image/svg+xml;utf8,
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900'>
        <defs>
            <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
                <stop offset='0%' stop-color='%2303124b'/>
                <stop offset='100%' stop-color='%234b1c7a'/>
            </linearGradient>
            <pattern id='dots' x='0' y='0' width='40' height='40' patternUnits='userSpaceOnUse'>
                <circle cx='2' cy='2' r='1.5' fill='rgba(255,255,255,0.12)'/>
            </pattern>
        </defs>
        <rect width='1600' height='900' fill='url(%23bg)'/>
        <rect width='1600' height='900' fill='url(%23dots)'/>
        <path d='M0,620 C400,540 800,700 1200,620 1400,580 1600,640 1600,640 L1600,900 L0,900 Z'
              fill='rgba(255,255,255,0.06)'/>
    </svg>");
    background-size: cover;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PROFESSIONAL UI THEME (HIGH CONTRAST)
# -------------------------------------------------
st.markdown("""
<style>
* {
    font-family: 'Segoe UI', sans-serif;
}

/* HERO */
.hero {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    padding: 48px;
    border-radius: 22px;
    color: #ffffff;
    margin-bottom: 45px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

.hero-subtitle {
    font-size: 18px;
    margin-top: 10px;
    color: #e5e7eb;
}

.hero-tag {
    margin-top: 14px;
    font-size: 14px;
    color: #c7d2fe;
}

/* CARDS */
.card {
    background-color: rgba(255, 255, 255, 0.96);
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.28);
    margin-bottom: 26px;
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #1e293b;
}

.card-icon {
    font-size: 32px;
    margin-bottom: 8px;
    color: #0d6efd;
}

.card-text {
    font-size: 15px;
    color: #1f2937;
    line-height: 1.75;
}

.card-text ul li {
    margin-bottom: 6px;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #e5e7eb;
    margin-top: 55px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🚨 Crowd Anomaly Detection System</div>
    <div class="hero-subtitle">
        Intelligent monitoring of crowd behaviour using Swarm Intelligence & Deep Learning
    </div>
    <div class="hero-tag">
        🎓 Project • 🧠 AI-Driven • 🛡️ Public Safety
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FEATURE CARDS
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📊</div>
        <div class="card-title">Dataset Analysis</div>
        <div class="card-text">
            Evaluate pre-recorded crowd surveillance videos to analyze:
            <ul>
                <li>Crowd density variation over time</li>
                <li>Motion intensity and activity patterns</li>
                <li>Abnormal behaviour detection accuracy</li>
                <li>Frame-level temporal analytics</li>
            </ul>
            Ideal for <b>offline evaluation</b>, experimentation, and report generation.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🎥</div>
        <div class="card-title">Live Crowd Detection</div>
        <div class="card-text">
            Monitor live camera feeds with real-time intelligence:
            <ul>
                <li>Instant crowd anomaly detection</li>
                <li>Bounding boxes on dense crowd regions</li>
                <li>Live accuracy and status updates</li>
                <li>Crowd pulse visualization</li>
            </ul>
            Designed for <b>real-time surveillance</b> environments.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# SYSTEM ARCHITECTURE
# -------------------------------------------------
st.markdown("""
<div class="card">
    <div class="card-icon">🧠</div>
    <div class="card-title">System Architecture Overview</div>
    <div class="card-text">
        The system integrates advanced AI techniques for robust crowd anomaly detection:
        <ul>
            <li><b>RIWPSO</b> for optimized crowd region segmentation</li>
            <li><b>MobileNetV2</b> for deep spatial feature extraction</li>
            <li><b>Ensemble Learning</b> using SVM and Random Forest</li>
            <li><b>Rule-based fusion</b> for decision refinement</li>
            <li><b>Streamlit dashboards</b> for interactive visualization</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<div class="footer">
    🔍 Crowd Behaviour Analysis & Anomaly Detection System<br>
    Built using Python • OpenCV • TensorFlow • Streamlit
</div>
""", unsafe_allow_html=True)
