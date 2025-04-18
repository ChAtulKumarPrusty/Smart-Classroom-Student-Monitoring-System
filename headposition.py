import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load student images and labels
student_images_path = "student image"  # Folder containing student images
students = {}
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_student_images():
    images, labels = [], []
    label_id = 0
    for filename in os.listdir(student_images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(student_images_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_id)
                students[label_id] = os.path.splitext(filename)[0]  # Store student name
                label_id += 1

    if images:
        recognizer.train(images, np.array(labels))
        print("Face recognition model trained.")

load_student_images()

# Head pose estimation
def get_head_pose(image, face_landmarks, image_width, image_height):
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array([
        (face_landmarks[1].x * image_width, face_landmarks[1].y * image_height),
        (face_landmarks[152].x * image_width, face_landmarks[152].y * image_height),
        (face_landmarks[33].x * image_width, face_landmarks[33].y * image_height),
        (face_landmarks[263].x * image_width, face_landmarks[263].y * image_height),
        (face_landmarks[61].x * image_width, face_landmarks[61].y * image_height),
        (face_landmarks[291].x * image_width, face_landmarks[291].y * image_height)
    ], dtype=np.float64)

    focal_length = image_width
    camera_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector

# Attention detection
def detect_attention(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))

    yaw_threshold = 0.4
    pitch_threshold = 0.3

    return "Looking at Instructor" if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold else "Distracted"

# Store attention status in Excel
def update_excel(student_name, status):
    file_path = "attention.xlsx"

    # Load existing data or create new
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["Student Name", "Status"])

    # Append new record
    new_entry = pd.DataFrame([[student_name, status]], columns=["Student Name", "Status"])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save to Excel
    df.to_excel(file_path, index=False)
    print(f"Updated attendance record for {student_name}.")

# Main function
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                rotation_vector = get_head_pose(frame, face_landmarks.landmark, frame.shape[1], frame.shape[0])
                status = detect_attention(rotation_vector)

                # Identify student
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    label, confidence = recognizer.predict(face_roi)

                    if confidence < 80:  # Threshold for recognition
                        student_name = students.get(label, "Unknown")
                        update_excel(student_name, status)

                        # Draw bounding box and label
                        color = (0, 255, 0) if status == "Looking at Instructor" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{student_name}: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Student Attention Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
