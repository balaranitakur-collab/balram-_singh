import cv2
import os

folders = [
    "dataset/raw",
    "dataset/preprocessed",
    "dataset/segmented"
]

for folder in folders:
    print("\nChecking:", folder)
    
    if not os.path.exists(folder):
        print("Folder not found")
        continue
        
    files = os.listdir(folder)
    
    if len(files) == 0:
        print("No images found")
    else:
        print("Images found:", len(files))