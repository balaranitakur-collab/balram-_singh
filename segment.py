import cv2
import numpy as np
import os

input_folder = "dataset/preprocessed"
output_folder = "dataset/segmented"

os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(input_folder):
    print("Preprocessed folder not found.")
    exit()

files = os.listdir(input_folder)

if len(files) == 0:
    print("No images in preprocessed folder.")
    exit()

for file in files:

    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path, 0)   # Load as grayscale

    if img is None:
        print("Skipping invalid image:", file)
        continue

    img_blur = cv2.GaussianBlur(img, (9, 9), 2)

    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=20,
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:

        circles = np.uint16(np.around(circles))

        x, y, r = circles[0][0]

        mask = np.zeros_like(img)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        iris = cv2.bitwise_and(img, img, mask=mask)

        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, iris)

        print("Segmented:", file)

    else:
        print("No circle detected in:", file)

print("Segmentation Finished Successfully")