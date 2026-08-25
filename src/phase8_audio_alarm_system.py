import os
import time
import pickle
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import pygame

# =========================
# PATHS
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_fatigue_model.pth")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.pkl")
SOUNDS_DIR = os.path.join(PROJECT_ROOT, "sounds")

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 30
SMOOTHING_WINDOW = 20

# Audio delays (seconds)
DROWSY_AUDIO_DELAY = 1.0
YAWN_AUDIO_DELAY = 1.5
DISTRACT_AUDIO_DELAY = 2.0
AUDIO_COOLDOWN = 2.0

# Adaptive thresholds
YAWN_MAR_THRESHOLD = 0.55
YAWN_JAW_STRETCH = 0.06        # Jaw drop threshold
DISTRACT_YAW_THRESHOLD = 16.0     # Left / Right turn angle (Tuned for smooth left & right turns)
DISTRACT_NOSE_SHIFT = 0.07        # Up / Down look threshold
DROWSY_EAR_RATIO = 0.65            # Drowsy if EAR < baseline_ear * 0.65

CALIBRATION_FRAMES = 45            # ~1.5 sec

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
HEAD_IDS = [1, 152, 263, 33, 291, 61]

# Key 2D vertical landmarks
NOSE_TIP = 1
TOP_HEAD = 10
CHIN = 152

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)


class FatigueMLP(nn.Module):
    def __init__(self, input_dim=180, num_classes=4):
        super(FatigueMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def ear(eye):
    v1 = dist(eye[1], eye[5])
    v2 = dist(eye[2], eye[4])
    h = dist(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0


def mar(mouth):
    v1 = dist(mouth[1], mouth[7])
    v2 = dist(mouth[2], mouth[6])
    v3 = dist(mouth[3], mouth[5])
    h = dist(mouth[0], mouth[4])
    return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0.0


def get_pts(landmarks, ids, w, h):
    pts = []
    for i in ids:
        lm = landmarks.landmark[i]
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float64)


def rot_to_ypr(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    else:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def head_pose(landmarks, w, h, cam, dist_coef):
    img_pts = get_pts(landmarks, HEAD_IDS, w, h)
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, img_pts, cam, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    return rot_to_ypr(R)


def get_2d_nose_ratio(landmarks):
    """Calculates 2D vertical nose position relative to facial height."""
    nose_y = landmarks.landmark[NOSE_TIP].y
    top_y = landmarks.landmark[TOP_HEAD].y
    chin_y = landmarks.landmark[CHIN].y

    face_h = chin_y - top_y
    if face_h <= 0:
        return 0.5
    return (nose_y - top_y) / face_h


def get_jaw_drop_ratio(landmarks):
    """Calculates 2D distance from nose to chin."""
    nose_y = landmarks.landmark[NOSE_TIP].y
    chin_y = landmarks.landmark[CHIN].y
    top_y = landmarks.landmark[TOP_HEAD].y

    face_h = chin_y - top_y
    if face_h <= 0:
        return 0.35
    return (chin_y - nose_y) / face_h


def main():
    pygame.mixer.init()
    sound_low = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "alert_low.wav"))
    sound_med = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "alert_medium.wav"))
    sound_high = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "alert_high.wav"))

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    model = FatigueMLP(input_dim=180, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    pred_history = collections.deque(maxlen=SMOOTHING_WINDOW)

    calibration_ear_samples = []
    calibration_nose_samples = []
    calibration_jaw_samples = []
    baseline_ear = None
    baseline_nose_ratio = None
    baseline_jaw_ratio = None
    is_calibrated = False

    prev_time = 0
    final_state = "ALERT"
    confidence = 0.0

    state_start_time = None
    last_audio_play_time = 0.0
    previous_state = "ALERT"

    color_map = {
        "alert": (0, 255, 0),
        "drowsy": (0, 0, 255),
        "yawning": (0, 165, 255),
        "distracted": (0, 255, 255),
        "no face": (160, 160, 160),
        "calibrating": (0, 255, 255),
    }

    print("🤖 Precision Universal AI Engine Active!")
    print("   Look straight ahead for ~1.5 sec for calibration.")
    print("   Press 'r' = Recalibrate | Press 'q' = Quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        focal = w
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        dist_coef = np.zeros((4, 1), dtype=np.float64)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        avg_ear, mar_val, yaw, nose_shift, jaw_shift = 0.0, 0.0, 0.0, 0.0, 0.0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            le = get_pts(face, LEFT_EYE, w, h)
            re = get_pts(face, RIGHT_EYE, w, h)
            mo = get_pts(face, MOUTH, w, h)

            ear_l = ear(le)
            ear_r = ear(re)
            avg_ear = (ear_l + ear_r) / 2.0
            mar_val = mar(mo)
            yaw, pitch, roll = head_pose(face, w, h, cam, dist_coef)
            curr_nose_ratio = get_2d_nose_ratio(face)
            curr_jaw_ratio = get_jaw_drop_ratio(face)

            for p in np.vstack([le, re, mo]):
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)

            # -------- AUTO CALIBRATION --------
            if not is_calibrated:
                calibration_ear_samples.append(avg_ear)
                calibration_nose_samples.append(curr_nose_ratio)
                calibration_jaw_samples.append(curr_jaw_ratio)

                if len(calibration_ear_samples) >= CALIBRATION_FRAMES:
                    baseline_ear = float(np.median(calibration_ear_samples))
                    baseline_nose_ratio = float(np.median(calibration_nose_samples))
                    baseline_jaw_ratio = float(np.median(calibration_jaw_samples))
                    is_calibrated = True
                    print(f"✅ CALIBRATED | Baseline EAR = {baseline_ear:.3f}")
                final_state = "calibrating"
                confidence = 0.0
            else:
                nose_shift = abs(curr_nose_ratio - baseline_nose_ratio)
                jaw_shift = curr_jaw_ratio - baseline_jaw_ratio

                current_features = [ear_l, ear_r, mar_val, yaw, pitch, roll]
                frame_buffer.append(current_features)

                if len(frame_buffer) == SEQUENCE_LENGTH:
                    flat_features = np.array(frame_buffer, dtype=np.float32).reshape(1, -1)
                    scaled_features = scaler.transform(flat_features)
                    tensor_in = torch.tensor(scaled_features, dtype=torch.float32)

                    with torch.no_grad():
                        logits = model(tensor_in)
                        probs = torch.softmax(logits, dim=1).numpy()[0]

                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx] * 100.0)
                    raw_pred = label_encoder.inverse_transform([pred_idx])[0]

                    drowsy_cutoff = baseline_ear * DROWSY_EAR_RATIO

                    # --- YAWN CONDITION (Strictly requires mouth to open!) ---
                    is_yawning = (mar_val > YAWN_MAR_THRESHOLD) or (jaw_shift > YAWN_JAW_STRETCH and mar_val > 0.15)

                    # --- PRIORITY RULES ---
                    # 1. Eyes Closed -> DROWSY (Red - Highest Priority)
                    if avg_ear < drowsy_cutoff:
                        raw_pred = "drowsy"

                    # 2. Mouth Open -> YAWNING (Orange)
                    elif is_yawning:
                        raw_pred = "yawning"

                    # 3. Head Turned OR Up/Down Look -> DISTRACTED (Yellow)
                    elif abs(yaw) >= DISTRACT_YAW_THRESHOLD or nose_shift >= DISTRACT_NOSE_SHIFT:
                        raw_pred = "distracted"

                    # 4. Normal awake -> ALERT
                    else:
                        if raw_pred == "yawning" and not is_yawning:
                            raw_pred = "alert"
                        elif raw_pred == "drowsy" and avg_ear > drowsy_cutoff:
                            raw_pred = "alert"
                        elif raw_pred == "distracted" and (abs(yaw) < DISTRACT_YAW_THRESHOLD and nose_shift < DISTRACT_NOSE_SHIFT):
                            raw_pred = "alert"

                    pred_history.append(raw_pred)

                    most_common, freq = collections.Counter(pred_history).most_common(1)[0]
                    if freq >= 12:
                        final_state = most_common
                    else:
                        final_state = "alert"
                else:
                    final_state = "alert"
        else:
            final_state = "no face"
            confidence = 0.0
            pred_history.clear()

        # -------- AUDIO CONTROLLER --------
        now_time = time.time()
        if final_state != previous_state:
            state_start_time = now_time
            previous_state = final_state

        state_duration = (now_time - state_start_time) if state_start_time else 0.0

        if is_calibrated and (now_time - last_audio_play_time) > AUDIO_COOLDOWN:
            if final_state == "drowsy" and state_duration >= DROWSY_AUDIO_DELAY:
                sound_high.play()
                last_audio_play_time = now_time
            elif final_state == "yawning" and state_duration >= YAWN_AUDIO_DELAY:
                sound_med.play()
                last_audio_play_time = now_time
            elif final_state == "distracted" and state_duration >= DISTRACT_AUDIO_DELAY:
                sound_low.play()
                last_audio_play_time = now_time

        state_color = color_map.get(final_state, (180, 180, 180))

        # ================= UI =================
        cv2.rectangle(frame, (10, 10), (440, 180), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (440, 180), (90, 90, 90), 1)

        if not is_calibrated:
            cv2.putText(frame, "CALIBRATING NEW FACE...", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Look at camera: {len(calibration_ear_samples)}/{CALIBRATION_FRAMES}",
                        (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(frame, "Keep eyes OPEN and face forward", (25, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            drowsy_cut = baseline_ear * DROWSY_EAR_RATIO
            cv2.putText(frame, "UNIVERSAL ENGINE ACTIVE", (25, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Baseline EAR: {baseline_ear:.2f} | Cutoff: {drowsy_cut:.2f}",
                        (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar_val:.2f} | Yaw: {yaw:+.1f}",
                        (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, f"STATE: {final_state.upper()}", (25, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(frame, f"Duration: {state_duration:.1f}s | Conf: {confidence:.1f}%",
                        (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if final_state in ["drowsy", "yawning", "distracted"]:
            cv2.rectangle(frame, (0, h - 50), (w, h), state_color, -1)
            cv2.putText(frame, f"[!] ALARM: {final_state.upper()} ({state_duration:.1f}s) [!]",
                        (w // 2 - 230, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0
        prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "R = Recalibrate | Q = Quit", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Driver Fatigue Monitor - Phase 8 Universal Audio Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_calibrated = False
            baseline_ear = None
            baseline_nose_ratio = None
            baseline_jaw_ratio = None
            calibration_ear_samples.clear()
            calibration_nose_samples.clear()
            calibration_jaw_samples.clear()
            frame_buffer.clear()
            pred_history.clear()
            final_state = "calibrating"

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pygame.mixer.quit()
    print("👋 Phase 8 closed cleanly.")


if __name__ == "__main__":
    main()