import cv2
import numpy as np

cap = cv2.VideoCapture(0)

print("Press 's' to capture iris")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:

        eye = gray[y:y+h, x:x+w]

        # Preprocessing
        eye = cv2.resize(eye, (200, 200))
        eye_blur = cv2.GaussianBlur(eye, (9,9), 2)

        # Segmentation using Hough Circle
        circles = cv2.HoughCircles(
            eye_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=20,
            minRadius=20,
            maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x1, y1, r = circles[0][0]

            mask = np.zeros_like(eye)
            cv2.circle(mask, (x1, y1), r, 255, -1)

            iris = cv2.bitwise_and(eye, eye, mask=mask)

            cv2.imshow("Iris Output", iris)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()