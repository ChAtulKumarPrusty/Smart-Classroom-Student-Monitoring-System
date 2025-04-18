import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import os
from datetime import datetime
from collections import Counter

# Load the trained emotion detection model
emotion_model = tf.keras.models.load_model("emotion_detection_model.h5")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.9)

# Load student images and store encodings
student_images_path = "student image"
student_data = {}

for file in os.listdir(student_images_path):
    img_path = os.path.join(student_images_path, file)
    img = cv2.imread(img_path)
    student_data[file] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Initialize webcam
cap = cv2.VideoCapture(0)

excel_file = "emotion_attendance.xlsx"

# Ensure the file exists with correct columns
if not os.path.exists(excel_file) or os.stat(excel_file).st_size == 0:
    df = pd.DataFrame(columns=["Name", "Emotion", "Time"])
    df.to_excel(excel_file, index=False)
else:
    df = pd.read_excel(excel_file)
# Dictionary to track emotions for each student
emotion_tracking = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, box_width, box_height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                           int(bboxC.width * w), int(bboxC.height * h))

            # Extract the detected face
            face_roi = frame[y:y+box_height, x:x+box_width]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Resize for emotion model
            face_resized = cv2.resize(face_gray, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=0) / 255.0

            # Predict emotion
            emotion_pred = emotion_model.predict(face_resized)
            emotion = emotion_labels[np.argmax(emotion_pred)]

            # Compare with student images
            name = "Unknown"
            min_diff = float("inf")

            for student_name, student_img in student_data.items():
                student_img_resized = cv2.resize(student_img, (face_gray.shape[1], face_gray.shape[0]))
                diff = np.sum((face_gray - student_img_resized) ** 2)

                if diff < min_diff:  # Find the closest match
                    min_diff = diff
                    name = os.path.splitext(student_name)[0]  # Remove file extension

            # Track emotions for this student
            if name not in emotion_tracking:
                emotion_tracking[name] = []

            emotion_tracking[name].append(emotion)

            # If we have at least 10 detections, save the most common one
            if len(emotion_tracking[name]) >= 10:
                most_common_emotion = Counter(emotion_tracking[name]).most_common(1)[0][0]

                # Read existing records
                df = pd.read_excel(excel_file)

                # Check if student is already recorded with the same emotion
                if not ((df["Name"] == name) & (df["Emotion"] == most_common_emotion)).any():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_entry = pd.DataFrame([[name, most_common_emotion, timestamp]],
                                             columns=["Name", "Emotion", "Time"])
                    df = pd.concat([df, new_entry], ignore_index=True)
                    df.to_excel(excel_file, index=False)

                    # Reset tracking after saving
                    emotion_tracking[name] = []

            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Emotion Detection & Attendance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
