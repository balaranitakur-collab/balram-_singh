import cv2
import numpy as np

def extract_features(iris_img):
    iris_resized = cv2.resize(iris_img, (64, 64))
    feature_vector = iris_resized.flatten()
    feature_vector = feature_vector / 255.0
    return feature_vector