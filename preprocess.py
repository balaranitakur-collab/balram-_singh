import cv2
import os

base_path = os.getcwd()

input_folder = os.path.join(base_path, "raw")
output_folder = os.path.join(base_path, "dataset", "preprocessed")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

files = os.listdir(input_folder)

if len(files) == 0:
    print("No images inside dataset/raw folder.")
    exit()

for file in files:
    img_path = os.path.join(input_folder, file)

    # Process only image files
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img = cv2.imread(img_path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    equalized = cv2.equalizeHist(blur)

    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, equalized)

    print("Processed:", file)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    equalized = cv2.equalizeHist(blur)

    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, equalized)

    print("Processed:", file)

print("Preprocessing Completed Successfully")