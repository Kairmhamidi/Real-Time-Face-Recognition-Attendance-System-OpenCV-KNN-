import cv2
import os
import numpy as np
import csv
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# ================= SETTINGS =================
ATTENDANCE_INTERVAL = 120      
DISTANCE_THRESHOLD = 3500      

# ================= CAMERA =================
video = cv2.VideoCapture(0)
if not video.isOpened():
    print(" Camera not accessible")
    exit()

# ================= FACE DETECTOR =================
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_detect.empty():
    print(" Haarcascade not loaded")
    exit()

# ================= LOAD DATA =================
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

min_len = min(len(FACES), len(LABELS))
FACES = FACES[:min_len]
LABELS = LABELS[:min_len]

# ================= IMAGE SIZE =================
feature_size = FACES.shape[1]
img_size = int(np.sqrt(feature_size))
if img_size * img_size != feature_size:
    print(" Feature size mismatch")
    exit()

print(f" Model image size: {img_size}x{img_size}")

# ================= TRAIN KNN =================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# ================= ATTENDANCE =================
COL_NAMES = ['Name', 'Time']
last_attendance_time = {}
os.makedirs('Attendance', exist_ok=True)

print("🎥 Attendance system running — Press 'q' to exit")

# ================= MAIN LOOP =================
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (img_size, img_size))
        face_flat = face_img.flatten().reshape(1, -1)

        # 🔥 GET DISTANCE
        distances, indices = knn.kneighbors(face_flat, n_neighbors=1)
        distance = distances[0][0]
        predicted_name = knn.predict(face_flat)[0]

        current_time = time.time()

        # ================= UNKNOWN PERSON =================
        if distance > DISTANCE_THRESHOLD:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, "UNKNOWN", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            continue

        # ================= KNOWN PERSON =================
        name = predicted_name

        if name not in last_attendance_time:
            last_attendance_time[name] = 0

        can_mark = (current_time - last_attendance_time[name]) >= ATTENDANCE_INTERVAL

        color = (0,255,0) if can_mark else (0,255,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{name}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if can_mark:
            last_attendance_time[name] = current_time

            date = datetime.fromtimestamp(current_time).strftime('%d-%m-%Y')
            time_now = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
            file_path = f'Attendance/Attendance_{date}.csv'
            exists = os.path.isfile(file_path)

            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(COL_NAMES)
                writer.writerow([name, time_now])

            print(f"✅ Attendance marked: {name} at {time_now}")

        else:
            remaining = int(ATTENDANCE_INTERVAL - (current_time - last_attendance_time[name]))
            cv2.putText(frame, f"Wait {remaining}s", (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("✅ Program Ended")
