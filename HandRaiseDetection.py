import cv2
import os
import mediapipe as mp
import pandas as pd
import numpy as np

# Paths
IMAGE_FOLDER = "student image"
EXCEL_FILE = "attendance.xlsx"

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load OpenCV DNN Face Detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
)

# Load student images and compute ORB descriptors
orb = cv2.ORB_create()
known_faces = {}
for file in os.listdir(IMAGE_FOLDER):
    if file.endswith(("jpg", "png", "jpeg")):
        path = os.path.join(IMAGE_FOLDER, file)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            known_faces[file.split(".")[0]] = descriptors

# Open webcam
cap = cv2.VideoCapture(0)

# Dictionary to store attendance data
attendance_data = {name: "Not Raised" for name in known_faces.keys()}

# Start detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=100) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        # Face Recognition
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face_roi = gray_frame[y1:y2, x1:x2]
                keypoints, descriptors = orb.detectAndCompute(face_roi, None)

                # Match with known faces
                name = "Unknown"
                best_matches = 0
                for student_name, known_descriptors in known_faces.items():
                    if descriptors is not None:
                        matches = bf.match(known_descriptors, descriptors)
                        if len(matches) > best_matches:
                            best_matches = len(matches)
                            name = student_name

                # Draw face bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Hand Detection
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                        index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                        # Check if hand is raised
                        if index_tip_y < wrist_y:
                            attendance_data[name] = "Raised"
                            cv2.putText(frame, "Hand Raised!", (x1, y2 + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Draw hand landmarks
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show frame
        cv2.imshow("Hand Raise & Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save data to Excel
df = pd.DataFrame(list(attendance_data.items()), columns=["Name", "Hand Raised"])
if os.path.exists(EXCEL_FILE):
    df.to_excel(EXCEL_FILE, index=False, mode='a', header=False)  # Append data
else:
    df.to_excel(EXCEL_FILE, index=False)  # Create new file

print(f"Attendance saved in {EXCEL_FILE}")
