import cv2
import time
import numpy as np
import mediapipe as mp

# ---- MediaPipe landmark indices we need ----
# Nose tip, chin, left eye outer corner, right eye outer corner,
# left mouth corner, right mouth corner
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 263,
    "right_eye_outer": 33,
    "left_mouth": 291,
    "right_mouth": 61,
}

# Approximate 3D model points of a generic face (in mm-like units)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye outer corner
    (43.3, 32.7, -26.0),      # right eye outer corner
    (-28.9, -28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1),     # right mouth corner
], dtype=np.float64)

YAW_DISTRACT_THRESHOLD = 25.0   # degrees


def extract_image_points(face_landmarks, img_w, img_h):
    """Get 2D pixel positions of the 6 key landmarks."""
    points = []
    for key in ["nose_tip", "chin", "left_eye_outer",
                "right_eye_outer", "left_mouth", "right_mouth"]:
        lm = face_landmarks.landmark[LANDMARK_IDS[key]]
        x = lm.x * img_w
        y = lm.y * img_h
        points.append((x, y))
    return np.array(points, dtype=np.float64)


def rotation_matrix_to_euler_angles(R):
    """
    Convert 3x3 rotation matrix to Yaw, Pitch, Roll (in degrees).
    Simple explanation: breaks the 3D rotation into 3 readable angles.
    """
    # sy helps detect gimbal-lock edge cases
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        # Fallback for rare extreme angles
        yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    return yaw, pitch, roll


def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, nose_point):
    """Draw X (red), Y (green), Z (blue) axes from the nose tip."""
    axis_points = np.array([
        (0.0, 0.0, 0.0),      # origin (nose)
        (50.0, 0.0, 0.0),     # X axis
        (0.0, 50.0, 0.0),     # Y axis
        (0.0, 0.0, 50.0),     # Z axis
    ], dtype=np.float64)

    img_points, _ = cv2.projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    img_points = img_points.reshape(-1, 2).astype(int)

    origin = tuple(img_points[0])
    cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 2)   # X - Red
    cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 2)   # Y - Green
    cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 2)   # Z - Blue


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

    prev_time = 0
    print("✅ Head Pose Engine running. Press 'q' to quit.")
    print("   Turn your head left/right/up/down to test Yaw & Pitch.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Simple camera matrix (no fancy calibration needed)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        yaw = pitch = roll = 0.0
        status = "NO FACE"
        status_color = (180, 180, 180)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            image_points = extract_image_points(face_landmarks, w, h)

            # solvePnP = find rotation & translation of the face
            success_pnp, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                # Convert rotation vector → matrix → yaw/pitch/roll
                R, _ = cv2.Rodrigues(rvec)
                yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

                # Draw the 3D axes on the nose
                draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, image_points[0])

                # Draw the 6 key points
                for pt in image_points.astype(int):
                    cv2.circle(frame, tuple(pt), 3, (0, 255, 255), -1)

                # Distraction check (Yaw only for now)
                if abs(yaw) > YAW_DISTRACT_THRESHOLD:
                    status = "DISTRACTED"
                    status_color = (0, 0, 255)
                else:
                    status = "LOOKING FORWARD"
                    status_color = (0, 255, 0)

        # ---- Telemetry panel ----
        cv2.rectangle(frame, (10, 10), (340, 175), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (340, 175), (80, 80, 80), 1)

        cv2.putText(frame, f"Yaw:   {yaw:+.1f} deg", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {pitch:+.1f} deg", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Roll:  {roll:+.1f} deg", (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Status: {status}", (25, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Driver Fatigue Monitor - Phase 3 Head Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("👋 Session closed cleanly.")


if __name__ == "__main__":
    main()