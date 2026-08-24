import os
import time
import pickle
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_fatigue_model.pth")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.pkl")

# Configuration
SEQUENCE_LENGTH = 30       # Feature buffer size (1 second)
SMOOTHING_WINDOW = 20      # Requires agreement across history before state change

# KHAWAJA'S CUSTOM FACE THRESHOLDS
EAR_DROWSY_HARD_CAP = 0.16  # Eyes MUST be below 0.16 to even consider DROWSY
MAR_YAWN_HARD_CAP = 0.58    # Mouth MUST be above 0.58 to even consider YAWNING
YAW_DISTRACT_CAP = 22.0     # Yaw MUST be > 22 deg to consider DISTRACTED

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
HEAD_IDS = [1, 152, 263, 33, 291, 61]

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


def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
        print("❌ Missing model files. Run Phase 6 first.")
        return

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    model = FatigueMLP(input_dim=180, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH))
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

    prev_time = 0
    final_state = "ALERT"
    confidence = 0.0

    color_map = {
        "alert": (0, 255, 0),        # Green
        "drowsy": (0, 0, 255),       # Red
        "yawning": (0, 165, 255),    # Orange
        "distracted": (0, 255, 255)  # Yellow
    }

    print("🤖 Precision-Tuned AI Fatigue Engine Running!")
    print("   Press 'q' to exit.")

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

        # Default telemetry values for safety
        avg_ear = 0.0
        mar_val = 0.0
        yaw = 0.0

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

            current_features = [ear_l, ear_r, mar_val, yaw, pitch, roll]
            frame_buffer.append(current_features)

            # Draw facial dots
            for p in np.vstack([le, re, mo]):
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)

            # Inference
            if len(frame_buffer) == SEQUENCE_LENGTH:
                flat_features = np.array(frame_buffer).reshape(1, -1)
                scaled_features = scaler.transform(flat_features)

                tensor_in = torch.tensor(scaled_features, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(tensor_in)
                    probs = torch.softmax(logits, dim=1).numpy()[0]

                pred_idx = np.argmax(probs)
                raw_conf = probs[pred_idx] * 100.0
                raw_pred = label_encoder.inverse_transform([pred_idx])[0]

                # STRICT PHYSICAL HARD-GATES (Overrides AI false alarms)
                if raw_pred == "drowsy" and avg_ear > EAR_DROWSY_HARD_CAP:
                    raw_pred = "alert"
                elif raw_pred == "yawning" and mar_val < MAR_YAWN_HARD_CAP:
                    raw_pred = "alert"
                elif raw_pred == "distracted" and abs(yaw) < YAW_DISTRACT_CAP:
                    raw_pred = "alert"

                pred_history.append(raw_pred)
                confidence = raw_conf

                # Majority voting over 20 frames
                counts = collections.Counter(pred_history)
                most_common, frequency = counts.most_common(1)[0]

                # Require at least 12 out of 20 frames to agree
                if frequency >= 12:
                    final_state = most_common
                else:
                    final_state = "ALERT"

        else:
            final_state = "NO FACE"
            confidence = 0.0
            pred_history.clear()

        state_color = color_map.get(final_state, (180, 180, 180))

        # ================= UI OVERLAY =================
        cv2.rectangle(frame, (10, 10), (390, 160), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (390, 160), (90, 90, 90), 1)

        cv2.putText(frame, "AI ENGINE: CALIBRATED HYBRID", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar_val:.2f}",
                    (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, f"STATE: {final_state.upper()}", (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Big Warning Bar
        if final_state in ["drowsy", "yawning", "distracted"]:
            cv2.rectangle(frame, (0, h - 50), (w, h), state_color, -1)
            cv2.putText(frame, f"[!] WARNING: {final_state.upper()} DETECTED [!]",
                        (w // 2 - 220, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0
        prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Driver Fatigue Monitor - Phase 7 Precision AI Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()