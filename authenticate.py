import cv2
import numpy as np
from feature_extraction import extract_features

stored_features = np.load("database/features.npy")

cap = cv2.VideoCapture(0)

print("Press 'a' to authenticate")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (200, 200))

        features = extract_features(eye)

        distance = np.linalg.norm(stored_features - features)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            if distance < 10:
                print("ACCESS GRANTED")
            else:
                print("ACCESS DENIED")

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Authenticate Iris", frame)

cap.release()
cv2.destroyAllWindows()