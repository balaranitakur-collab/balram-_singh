import cv2
import numpy as np
import os
from feature_extraction import extract_features

os.makedirs("database", exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press 's' to enroll iris")
print("Press 'q' to quit")

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (200, 200))

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Enroll Iris", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            features = extract_features(eye)
            np.save("database/features.npy", features)
            print("Iris Enrolled Successfully")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Enroll Iris", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()