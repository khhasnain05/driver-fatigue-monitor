import cv2
import time
import mediapipe as mp

def main():
    # 1. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # refine_landmarks=True enables detailed eye/iris tracking points
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Frame rate calculation variables
    prev_time = 0

    print("✅ Webcam started successfully. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Warning: Empty frame received from webcam.")
            break

        # Flip horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe requires RGB format, OpenCV gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # 3. Draw face landmarks if a face is detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh lines
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw eye and lips contours
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # 4. Calculate and display FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw FPS on screen
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # 5. Display the output window
        cv2.imshow("Driver Fatigue Monitor - Phase 1 Baseline", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("👋 Session closed cleanly.")

if __name__ == "__main__":
    main()