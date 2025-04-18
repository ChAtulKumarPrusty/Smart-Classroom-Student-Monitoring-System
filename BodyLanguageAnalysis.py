import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from openpyxl import Workbook

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Path to student images folder
student_images_path = "student image"

# Load student images and their names
known_face_data = []
known_face_names = []

for filename in os.listdir(student_images_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(student_images_path, filename)
        img = cv2.imread(image_path)
        known_face_data.append(img)
        known_face_names.append(os.path.splitext(filename)[0])  # Extract name

# Function to check head position for inattentiveness
def detect_head_tilt(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    shoulder_mid = (left_shoulder.x + right_shoulder.x) / 2
    head_tilt = abs(nose.x - shoulder_mid)

    return "Inattentive" if head_tilt > 0.1 else "Attentive"

# Function to match face with known student images
def recognize_face(frame, known_faces):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_match_name = "Unknown"
    best_match_score = float("inf")

    for idx, known_face in enumerate(known_faces):
        gray_known_face = cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, gray_known_face, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > 0.6 and max_val < best_match_score:  # Higher score means better match
            best_match_name = known_face_names[idx]
            best_match_score = max_val

    return best_match_name

# Excel file setup
excel_file = "student_attendance.xlsx"

# Create a new Excel file if it does not exist
if not os.path.exists(excel_file):
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Attention"])
    wb.save(excel_file)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        student_name = recognize_face(face_roi, known_face_data)

        # Pose detection for head tilt (attentiveness check)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        attention_status = "Unknown"

        if results.pose_landmarks:
            attention_status = detect_head_tilt(results.pose_landmarks.landmark)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display face box and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{student_name}: {attention_status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update Excel file
        df = pd.read_excel(excel_file)
        new_data = pd.DataFrame([[student_name, attention_status]], columns=["Name", "Attention"])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_excel(excel_file, index=False)

    cv2.imshow("Student Attention Tracker", frame)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
