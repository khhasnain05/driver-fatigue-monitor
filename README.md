# 🚗 Real-Time Driver Fatigue, Drowsiness & Attention Monitoring System

An AI-powered computer vision application that monitors a driver's face in real time to detect drowsiness, yawning, and visual distraction using MediaPipe FaceMesh, custom geometric features, PyTorch, and multi-level audio alerts.

---

## 🌟 Key Features

* **Passive Driver Monitoring:** No physical strain, operates automatically via webcam.
* **Auto-Calibration Engine:** Learns any driver's facial structure in 1.5 seconds on startup.
* **PyTorch Deep Learning Engine:** Classifies driver behavior across 4 primary states (`ALERT`, `DROWSY`, `YAWNING`, `DISTRACTED`).
* **Multi-Level Audio Alarms:** Plays distinct warning sounds based on risk level:
  * 🔴 **Level 3 (Red):** Closed eyes ➔ High emergency siren.
  * 🟠 **Level 2 (Orange):** Yawning ➔ Double pulse warning.
  * 🟡 **Level 1 (Yellow):** Looking away/down ➔ Soft notice beep.
* **Hand-Occlusion Handling:** Immune to false distraction alarms when covering a yawn with a hand.
* **Low Hardware Footprint:** Runs at 28-31 FPS on CPU (tested on Intel i7-1355U, 8 GB RAM, no GPU).

---

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Vision:** OpenCV, MediaPipe FaceMesh (468 3D landmarks)
* **Deep Learning:** PyTorch (CPU-only build)
* **Audio:** Pygame Mixer
* **Data Math:** NumPy, Scikit-Learn, Pandas

---

## 📁 Project Structure

```text
driver-fatigue-monitor/
├── data/
│   ├── landmark_features/    # Recorded CSV dataset sequences
│   └── processed/            # Preprocessed .npz datasets
├── models/
│   ├── best_fatigue_model.pth# Trained PyTorch MLP weights
│   ├── scaler.pkl            # StandardScaler weights
│   └── label_encoder.pkl     # Class encoder
├── sounds/                   # Alert audio files (.wav)
├── src/
│   ├── phase1_face_mesh_baseline.py
│   ├── phase2_eye_mar_geometry.py
│   ├── phase3_head_pose_estimation.py
│   ├── phase4_rule_based_detector.py
│   ├── phase5_dataset_collector.py
│   ├── verify_dataset.py
│   ├── phase6_preprocess_dataset.py
│   ├── phase6_train_fatigue_model.py
│   ├── generate_sounds.py
|   ├── phase7_dl_fatigue_engine.py
│   └── phase8_audio_alarm_system.py   # Main Application Script
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

**1. Setup Virtual Environment:**
```bash
git clone https://github.com/khhasnain05/driver-fatigue-monitor.git
cd driver-fatigue-monitor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Generate Sound Files (One-time setup):**
```bash
python src/generate_sounds.py
```

**3. Run the Main Application:**
```bash
python src/phase8_audio_alarm_system.py
```

---

## 🕹️ Controls

* Press `R` : Recalibrate system for a new driver
* Press `Q` : Quit application cleanly