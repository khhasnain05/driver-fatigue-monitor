import cv2
import time
import numpy as np
import mediapipe as mp

# =========================
# THRESHOLDS (tune later)
# =========================
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.65
YAW_THRESHOLD = 25.0          # degrees

DROWSY_FRAMES_NEEDED = 30     # ~1 sec at 30 FPS
YAWN_FRAMES_NEEDED = 15       # ~0.5 sec
DISTRACT_FRAMES_NEEDED = 60   # ~2 sec

# =========================
# LANDMARK INDICES
# =========================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

# Head-pose landmarks: nose, chin, left eye outer, right eye outer, left mouth, right mouth
HEAD_IDS = [1, 152, 263, 33, 291, 61]

# 3D model points for head pose (generic face)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def eye_aspect_ratio(eye):
    v1 = dist(eye[1], eye[5])
    v2 = dist(eye[2], eye[4])
    h = dist(eye[0], eye[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def mouth_aspect_ratio(mouth):
    v1 = dist(mouth[1], mouth[7])
    v2 = dist(mouth[2], mouth[6])
    v3 = dist(mouth[3], mouth[5])
    h = dist(mouth[0], mouth[4])
    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def get_pts(landmarks, indices, w, h):
    pts = []
    for i in indices:
        lm = landmarks.landmark[i]
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float64)


def rotation_to_yaw_pitch_roll(R):
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


def estimate_head_pose(landmarks, w, h, camera_matrix, dist_coeffs):
    image_points = get_pts(landmarks, HEAD_IDS, w, h)
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0, None, None
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotation_to_yaw_pitch_roll(R)
    return yaw, pitch, roll, rvec, tvec


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Counters (how many frames the condition stayed true)
    drowsy_counter = 0
    yawn_counter = 0
    distract_counter = 0

    prev_time = 0
    print("✅ Rule-Based Fatigue Detector running.")
    print("   Test: close eyes, yawn, look away. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Simple camera matrix (no calibration needed)
        focal = w
        cam_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        avg_ear = 0.0
        mar = 0.0
        yaw = pitch = roll = 0.0
        state = "NO FACE"
        state_color = (160, 160, 160)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # --- Geometry ---
            left_eye = get_pts(face, LEFT_EYE, w, h)
            right_eye = get_pts(face, RIGHT_EYE, w, h)
            mouth = get_pts(face, MOUTH, w, h)

            ear_l = eye_aspect_ratio(left_eye)
            ear_r = eye_aspect_ratio(right_eye)
            avg_ear = (ear_l + ear_r) / 2.0
            mar = mouth_aspect_ratio(mouth)

            yaw, pitch, roll, rvec, tvec = estimate_head_pose(
                face, w, h, cam_matrix, dist_coeffs
            )

            # Draw eye / mouth points
            eye_col = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
            mouth_col = (0, 0, 255) if mar > MAR_THRESHOLD else (0, 255, 255)
            for p in np.vstack([left_eye, right_eye]):
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, eye_col, -1)
            for p in mouth:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, mouth_col, -1)

            # --- Frame counters (temporal filter) ---
            # Eyes closed?
            if avg_ear < EAR_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            # Yawning?
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            # Looking away?
            if abs(yaw) > YAW_THRESHOLD:
                distract_counter += 1
            else:
                distract_counter = 0

            # --- State machine with priority ---
            if drowsy_counter >= DROWSY_FRAMES_NEEDED:
                state = "DROWSY"
                state_color = (0, 0, 255)          # Red
            elif yawn_counter >= YAWN_FRAMES_NEEDED:
                state = "YAWNING"
                state_color = (0, 165, 255)        # Orange
            elif distract_counter >= DISTRACT_FRAMES_NEEDED:
                state = "DISTRACTED"
                state_color = (0, 255, 255)        # Yellow
            else:
                state = "ALERT"
                state_color = (0, 255, 0)          # Green

        # ========== UI PANEL ==========
        cv2.rectangle(frame, (10, 10), (370, 250), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (370, 250), (90, 90, 90), 1)

        cv2.putText(frame, f"EAR:  {avg_ear:.2f}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR:  {mar:.2f}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Yaw:  {yaw:+.1f} deg", (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        cv2.putText(frame, f"Drowsy frames:    {drowsy_counter}/{DROWSY_FRAMES_NEEDED}",
                    (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Yawn frames:      {yawn_counter}/{YAWN_FRAMES_NEEDED}",
                    (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Distract frames:  {distract_counter}/{DISTRACT_FRAMES_NEEDED}",
                    (25, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Big status badge
        cv2.putText(frame, f"STATE: {state}", (25, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)

        # Big on-screen warning when not alert
        if state in ("DROWSY", "YAWNING", "DISTRACTED"):
            cv2.rectangle(frame, (0, h - 60), (w, h), state_color, -1)
            cv2.putText(frame, f"! {state} !", (w // 2 - 120, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0
        prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Driver Fatigue Monitor - Phase 4 Rule Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("👋 Session closed cleanly.")


if __name__ == "__main__":
    main()