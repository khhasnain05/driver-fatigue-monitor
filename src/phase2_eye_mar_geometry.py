import cv2
import time
import numpy as np
import mediapipe as mp

# Landmark Indices from MediaPipe FaceMesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

# Baseline Thresholds (we can tune these later)
EAR_CLOSED_THRESHOLD = 0.21
MAR_YAWN_THRESHOLD = 0.65


def distance_2d(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return np.linalg.norm(p1 - p2)


def calculate_ear(landmarks_2d):
    """
    landmarks_2d: 6 points of the eye in order:
    [p1(outer), p2(top-left), p3(top-right), p4(inner), p5(bottom-right), p6(bottom-left)]
    """
    # Vertical distances
    v1 = distance_2d(landmarks_2d[1], landmarks_2d[5])
    v2 = distance_2d(landmarks_2d[2], landmarks_2d[4])
    # Horizontal distance
    h = distance_2d(landmarks_2d[0], landmarks_2d[3])

    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calculate_mar(landmarks_2d):
    """
    landmarks_2d: 8 points around inner/outer lips:
    [left_corner, upper_1, upper_mid, upper_2, right_corner, lower_2, lower_mid, lower_1]
    """
    # Vertical distances across upper and lower lip
    v1 = distance_2d(landmarks_2d[1], landmarks_2d[7])
    v2 = distance_2d(landmarks_2d[2], landmarks_2d[6])
    v3 = distance_2d(landmarks_2d[3], landmarks_2d[5])
    # Horizontal distance between mouth corners
    h = distance_2d(landmarks_2d[0], landmarks_2d[4])

    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def extract_pixel_coords(face_landmarks, indices, img_w, img_h):
    """Extracts (x, y) pixel coordinates for a specific list of indices."""
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        coords.append([x, y])
    return np.array(coords, dtype=np.float32)


def main():
    # 1. Initialize MediaPipe Face Mesh
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
    print("✅ Geometry Engine running. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmark coordinates
                left_eye = extract_pixel_coords(face_landmarks, LEFT_EYE_INDICES, w, h)
                right_eye = extract_pixel_coords(face_landmarks, RIGHT_EYE_INDICES, w, h)
                mouth = extract_pixel_coords(face_landmarks, MOUTH_INDICES, w, h)

                # Calculate EAR & MAR
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                avg_ear = (ear_left + ear_right) / 2.0
                mar = calculate_mar(mouth)

                # 2. Draw eye landmarks (green if open, red if closed)
                eye_color = (0, 0, 255) if avg_ear < EAR_CLOSED_THRESHOLD else (0, 255, 0)
                for pt in np.vstack([left_eye, right_eye]):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, eye_color, -1)

                # 3. Draw mouth landmarks (yellow if normal, red if yawning)
                mouth_color = (0, 0, 255) if mar > MAR_YAWN_THRESHOLD else (0, 255, 255)
                for pt in mouth:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, mouth_color, -1)

                # 4. Display telemetry panel on top left
                # Background card for readability
                cv2.rectangle(frame, (10, 10), (330, 180), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, 10), (330, 180), (80, 80, 80), 1)

                # Text readouts
                cv2.putText(frame, f"EAR Left:  {ear_left:.2f}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"EAR Right: {ear_right:.2f}", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Avg EAR:   {avg_ear:.2f}", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                cv2.putText(frame, f"MAR:       {mar:.2f}", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)

                # Eye state badge
                eye_status = "CLOSED" if avg_ear < EAR_CLOSED_THRESHOLD else "OPEN"
                m_status = "YAWN" if mar > MAR_YAWN_THRESHOLD else "NORMAL"
                cv2.putText(frame, f"Eyes: {eye_status}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                cv2.putText(frame, f"Mouth: {m_status}", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Driver Fatigue Monitor - Phase 2 Geometry", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()