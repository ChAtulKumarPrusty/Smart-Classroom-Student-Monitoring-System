# 🎓 Smart Classroom Student Monitoring System

This project is a **Real-Time AI-Powered Classroom Monitoring System** that tracks and evaluates student behaviors such as hand-raising, head position, emotion, and body posture using computer vision and deep learning techniques. It aims to provide automated attendance and attention analysis during online or physical classroom sessions.

## 🧠 Features

### 📌 1. Hand Raise Detection (`HandRaiseDetection.py`)
- Detects if a student has raised their hand during class.
- Recognizes faces using ORB descriptors and OpenCV's DNN Face Detector.
- Attendance is marked in `attendance.xlsx` with hand-raising status.

### 📌 2. Head Position & Attention Detection (`headposition.py`)
- Uses face mesh and pose estimation to determine if the student is looking at the instructor or distracted.
- Trains a face recognizer using student images.
- Logs attention status to `attention.xlsx`.

### 📌 3. Emotion Detection (`emotion.py`)
- Recognizes real-time facial emotions using a pre-trained CNN (`emotion_detection_model.h5`).
- Identifies students by comparing detected faces with saved student images.
- Logs the most frequently detected emotion in `emotion_attendance.xlsx`.

### 📌 4. Body Language Analysis (`BodyLanguageAnalysis.py`)
- Tracks student posture and detects head tilt using MediaPipe Pose.
- Recognizes students and logs their attentiveness to `student_attendance.xlsx`.

---

## 🛠️ Requirements

- Python 3.7+
- Required Libraries:
  pip install opencv-python mediapipe tensorflow pandas openpyxl


🗂️ Folder Structure
project/
│
├── HandRaiseDetection.py
├── headposition.py
├── emotion.py
├── BodyLanguageAnalysis.py
├── student image/              # Folder containing student face images
├── emotion_detection_model.h5 # Trained emotion model (required for emotion.py)
├── attendance.xlsx             # Generated: Hand raise log
├── attention.xlsx              # Generated: Head position log
├── emotion_attendance.xlsx     # Generated: Emotion log
└── student_attendance.xlsx     # Generated: Posture-based attention log

🖼️ Student Images
Place individual student face images inside the student image folder. Image file names should be the student's name (e.g., john_doe.jpg).

 📦 How to Run
Each script can be run independently based on the feature you'd like to use:
python HandRaiseDetection.py
python headposition.py
python emotion.py
python BodyLanguageAnalysis.py

Press q to stop webcam and save logs.

🔍 Use Cases
Real-time classroom engagement tracking.
Automated attendance and behavior logging.
Emotion monitoring for student well-being.
Instructor feedback and personalized intervention.

👨‍💻 Author
Ch Atul Kumar Prusty
📷 Linkedin: [https://www.linkedin.com/in/chatulkumarprusty/]
📧 Email: [chatulprusty@gmail.com]


---

Let me know if you want me to include:
- GIFs/screenshots for each feature
- Instructions for training your own emotion model
- A `requirements.txt` or `setup.py` file

Want me to generate the `requirements.txt` for you too?
