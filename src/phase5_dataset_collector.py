import cv2
import time
import csv
import os
from datetime import datetime
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 30          # frames per sample
COUNTDOWN_SECS = 2            # seconds to prepare before recording
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "landmark_features")
FPS_TARGET_HINT = 30

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

CLASSES = {
    ord("1"): "alert",
    ord("2"): "drowsy",
    ord("3"): "yawning",
    ord("4"): "distracted",
}


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


def extract_features(face, w, h, cam, dist_coef):
    """Returns [ear_l, ear_r, mar, yaw, pitch, roll]"""
    le = get_pts(face, LEFT_EYE, w, h)
    re = get_pts(face, RIGHT_EYE, w, h)
    mo = get_pts(face, MOUTH, w, h)
    yaw, pitch, roll = head_pose(face, w, h, cam, dist_coef)
    return [
        float(ear(le)),
        float(ear(re)),
        float(mar(mo)),
        float(yaw),
        float(pitch),
        float(roll),
    ]


def ensure_csv(path):
    """Create CSV with header if it does not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        header = ["sample_id", "label", "timestamp"]
        # 30 frames * 6 features
        for t in range(SEQUENCE_LENGTH):
            header += [
                f"f{t}_ear_l", f"f{t}_ear_r", f"f{t}_mar",
                f"f{t}_yaw", f"f{t}_pitch", f"f{t}_roll",
            ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def count_labels(path):
    counts = {"alert": 0, "drowsy": 0, "yawning": 0, "distracted": 0}
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = row.get("label", "")
            if lab in counts:
                counts[lab] += 1
    return counts


def draw_panel(frame, counts, mode, msg, progress=None):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (10, 10), (420, 230), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (420, 230), (90, 90, 90), 1)

    cv2.putText(frame, "DATASET COLLECTOR", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"alert:      {counts['alert']}", (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, f"drowsy:     {counts['drowsy']}", (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    cv2.putText(frame, f"yawning:    {counts['yawning']}", (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
    cv2.putText(frame, f"distracted: {counts['distracted']}", (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.putText(frame, "Keys: 1=alert 2=drowsy 3=yawn 4=distract | q=quit",
                (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Mode: {mode}", (25, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if msg:
        cv2.rectangle(frame, (0, h - 55), (w, h), (0, 140, 255), -1)
        cv2.putText(frame, msg, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Progress bar while recording
    if progress is not None:
        p = progress / SEQUENCE_LENGTH
        bar_w = int((w - 40) * p)
        cv2.rectangle(frame, (20, h - 80), (w - 20, h - 60), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 80), (20 + bar_w, h - 60), (0, 255, 0), -1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"fatigue_dataset_{stamp}.csv")
    ensure_csv(csv_path)

    counts = count_labels(csv_path)
    sample_id = sum(counts.values())

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    # idle | countdown | recording
    mode = "idle"
    active_label = None
    countdown_end = 0.0
    buffer = []

    print("✅ Dataset collector started.")
    print(f"📁 Saving to: {csv_path}")
    print("Keys: 1=alert  2=drowsy  3=yawning  4=distracted  q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        focal = w
        cam = np.array([[focal, 0, w / 2],
                        [0, focal, h / 2],
                        [0, 0, 1]], dtype=np.float64)
        dist_coef = np.zeros((4, 1), dtype=np.float64)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        features = None
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            features = extract_features(face, w, h, cam, dist_coef)

            # small face dots so user knows tracking is alive
            le = get_pts(face, LEFT_EYE, w, h)
            re = get_pts(face, RIGHT_EYE, w, h)
            mo = get_pts(face, MOUTH, w, h)
            for p in np.vstack([le, re, mo]):
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

        msg = ""
        progress = None
        now = time.time()

        # ---------- STATE MACHINE ----------
        if mode == "countdown":
            remaining = countdown_end - now
            if remaining > 0:
                msg = f"Get ready: {active_label.upper()} in {int(remaining) + 1}"
            else:
                mode = "recording"
                buffer = []
                msg = f"Recording {active_label.upper()}..."

        elif mode == "recording":
            if features is None:
                msg = "Face lost! Move back into frame..."
            else:
                buffer.append(features)
                progress = len(buffer)
                msg = f"Recording {active_label.upper()}... {len(buffer)}/{SEQUENCE_LENGTH}"

                if len(buffer) >= SEQUENCE_LENGTH:
                    # flatten 30 x 6 -> 180
                    flat = []
                    for row in buffer:
                        flat.extend(row)

                    ts = datetime.now().isoformat(timespec="seconds")
                    row_out = [sample_id, active_label, ts] + flat
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row_out)

                    counts[active_label] += 1
                    sample_id += 1
                    print(f"💾 Saved #{sample_id - 1} [{active_label}] | totals={counts}")

                    mode = "idle"
                    active_label = None
                    buffer = []
                    msg = "Saved! Press 1/2/3/4 for next sample."

        else:  # idle
            msg = "Press 1/2/3/4 to record a sample"

        draw_panel(frame, counts, mode, msg, progress)

        cv2.imshow("Driver Fatigue Monitor - Phase 5 Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Start a new sample only from idle (edge-triggered, no spam)
        if mode == "idle" and key in CLASSES:
            if results.multi_face_landmarks is None:
                print("⚠️ Face not found. Face the camera first.")
                continue
            active_label = CLASSES[key]
            mode = "countdown"
            countdown_end = time.time() + COUNTDOWN_SECS
            print(f"▶️ Starting sample for: {active_label}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print("\n========== SESSION SUMMARY ==========")
    print(f"File: {csv_path}")
    for k, v in counts.items():
        print(f"  {k:12s}: {v}")
    print(f"  TOTAL       : {sum(counts.values())}")
    print("=====================================")
    print("👋 Done. Safe testing only at your desk — never while driving.")


if __name__ == "__main__":
    main()