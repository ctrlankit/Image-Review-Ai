import cv2
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "clothes_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

## Load model
model = load_model(MODEL_PATH)

## Load correct class order
with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)

## Reverse dict: index -> class name
classes = {v: k for k, v in class_indices.items()}


def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "invalid_image", 0.0

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    if confidence < 0.50:
        return "unknown_image", confidence

    return classes[class_index], confidence
