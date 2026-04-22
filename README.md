Swarm Intelligence–Based Segmentation and Ensemble Learning for Abnormal Crowd Behaviour Detection
📌 Project Overview

This project presents a hybrid AI-based system for detecting abnormal crowd behaviour in surveillance videos by combining:

Swarm Intelligence (RIWPSO-based segmentation)
Deep CNN feature extraction (MobileNetV2)
Ensemble Machine Learning (SVM + Random Forest)
Rule-based decision fusion

The system is designed to improve robustness, efficiency, and real-time applicability by reducing background noise and optimizing crowd region detection.

❗ Problem Statement

Traditional crowd monitoring systems suffer from:

Low accuracy in dense crowd environments
High false positives due to background noise
Poor performance in real-time detection

This project addresses these issues using optimized segmentation + hybrid decision-making.

💡 Proposed System Architecture

The system follows a 6-step pipeline:

🔹 1. Video Input & Frame Extraction
Extracts frames from surveillance videos
Processes frames at a fixed interval for efficiency
🔹 2. Frame Preprocessing
Converts frames to grayscale
Resizes images to 224 × 224
Reduces noise and standardizes input
🔹 3. Swarm Intelligence-Based Segmentation (RIWPSO)
Uses Random Inertia Weight Particle Swarm Optimization
Optimizes threshold values for segmentation
Separates foreground crowd regions from background
Runs periodically (not every frame) for performance optimization
🔹 4. Feature Extraction
✔ CNN Features
Uses MobileNetV2 (pretrained on ImageNet)
Extracts 1280-dimensional feature vectors
✔ Motion Features (Supportive)
Uses Optical Flow (Farneback method)
Extracts:
Motion magnitude
Direction variance
Motion energy

👉 Note: Motion features are used for rule-based scoring, not primary ML training.

🔹 5. Hybrid Ensemble Learning
✔ Machine Learning Models:
Support Vector Machine (SVM)
Random Forest (RF)
Combined using Soft Voting Classifier
✔ Dataset Strategy:
Features are extracted from segmented images
Labels are automatically assigned (prototype-based):
First half → Normal
Second half → Abnormal
🔹 6. Decision Fusion & Anomaly Detection

The final decision is made using a weighted hybrid approach:

ML Score (SVM + RF probabilities)
Rule-Based Score (crowd density + motion)

Final classification:

NORMAL
ABNORMAL
⚙️ Key Innovation

✔ Hybrid Rule + ML Ensemble Voting
✔ RIWPSO optimization applied selectively (not every frame)
✔ Combination of spatial + motion intelligence
✔ Designed for real-time and offline analysis

🛠️ Tech Stack
Programming
Python
Libraries
OpenCV
NumPy
Scikit-learn
TensorFlow (MobileNetV2)
PyTorch (LSTM - experimental)
PySwarms
Tools
Visual Studio Code
Streamlit (UI Dashboard)

📂 Project Structure
Abnormal-Crowd-Detection
│
├── preprocessing/
│   ├── video_to_frames.py
│   ├── frame_preprocessing.py
│
├── swarm_segmentation/
│   ├── riwpso_segmentation.py
│
├── feature_extraction/
│   ├── cnn_features.py
│   ├── motion_features.py
│   ├── feature_fusion.py
│
├── ensemble_model/
│   ├── build_dataset.py
│   ├── ensemble_classifier.py
│   ├── ensemble_voter.py
│   ├── lstm_model.py   (experimental)
│
├── realtime/
│   ├── realtime_core.py
│
├── evaluation/
│   ├── metrics.py
│
├── app.py
├── main.py
├── requirements.txt
└── results/

▶️ How to Run the Project
Step 1 — Clone Repository
git clone https://github.com/loga-anand/Abnormal-Crowd-Detection-Using-Swarm-Based-Segmentation-.git
cd Abnormal-Crowd-Detection-Using-Swarm-Based-Segmentation-
Step 2 — Create Virtual Environment
python -m venv crowd_env
crowd_env\Scripts\activate
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Run Full Pipeline
python main.py
Step 5 — Run Streamlit App (Optional)
streamlit run app.py


📊 Evaluation Metrics

The system evaluates performance using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix

👉 Metrics are computed using scikit-learn evaluation functions.

📸 Output
Bounding boxes on detected crowd regions
Real-time anomaly labels (NORMAL / ABNORMAL)
Crowd density graphs
Motion intensity plots
📦 Dataset
Dataset is not included due to size limitations
Prototype dataset is automatically labeled during training
Can be replaced with real-world datasets (e.g., UCSD, UMN)
🔮 Future Enhancements
Use real labeled datasets for better accuracy
Fully integrate LSTM for temporal learning
Deploy on edge devices (CCTV systems)
Improve multi-crowd tracking and behavior prediction
Optimize model for low-latency real-time inference

👨‍💻 Author

Loga Anand M
Computer Science and Engineering 

⭐ Conclusion

This project demonstrates a practical and optimized approach for abnormal crowd behaviour detection by integrating:

Swarm Intelligence for segmentation
Deep learning for feature extraction
Ensemble learning for classification
Hybrid rule-based decision making

The system is suitable for smart surveillance, public safety, and real-time monitoring applications, with strong potential for further enhancement.
