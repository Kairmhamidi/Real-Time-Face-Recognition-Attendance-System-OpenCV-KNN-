import cv2
import os
import numpy as np
import pickle
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error loading Haar Cascade")
    exit()

os.makedirs("data", exist_ok=True)
face_data = []
count = 0
name = input("Enter the name of the person: ")
print(" Collecting face data... Look at the camera")

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]   
        face_img = cv2.resize(face_img, (50, 50))
        face_data.append(face_img.flatten())
        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{count}/100", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if count >= 100:
            break

    cv2.imshow("Face Dataset Creation", frame)
    if cv2.waitKey(1) == ord('q') or count >= 100:
        break

video.release()
cv2.destroyAllWindows()
faceData = np.array(face_data)  
if os.path.exists("data/names.pkl"):
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
else:
    names = []
names.extend([name] * 100)
with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

if os.path.exists("data/face_data.pkl"):
    with open("data/face_data.pkl", "rb") as f:
        existing_faces = pickle.load(f)
    faceData = np.vstack((existing_faces, faceData))

with open("data/face_data.pkl", "wb") as f:
    pickle.dump(faceData, f)

print(" Dataset saved successfully!")
print(f"Total faces: {faceData.shape[0]}")
print(f"Feature size per face: {faceData.shape[1]}")
