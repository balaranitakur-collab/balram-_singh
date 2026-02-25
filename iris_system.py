import cv2
import numpy as np
import os

# Create database folder if not exists
os.makedirs("database", exist_ok=True)

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

def extract_features(iris_img):
    iris_resized = cv2.resize(iris_img, (64, 64))
    feature_vector = iris_resized.flatten()
    feature_vector = feature_vector / 255.0
    return feature_vector

print("Press E to Enroll")
print("Press A to Authenticate")
print("Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (200, 200))

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        key = cv2.waitKey(1) & 0xFF

        # ENROLL
        if key == ord('e'):
            features = extract_features(eye)
            np.save("database/features.npy", features)
            print("Iris Enrolled Successfully")

        # AUTHENTICATE
        if key == ord('a'):
            if not os.path.exists("database/features.npy"):
                print("No enrolled user found!")
            else:
                stored_features = np.load("database/features.npy")
                features = extract_features(eye)
                distance = np.linalg.norm(stored_features - features)

                if distance < 10:
                    print("ACCESS GRANTED")
                    cv2.putText(frame, "ACCESS GRANTED", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    print("ACCESS DENIED")
                    cv2.putText(frame, "ACCESS DENIED", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Professional Iris Authentication System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()