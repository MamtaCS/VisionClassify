
from pathlib import Path
import sys
import cv2
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "result" / "best_model.joblib"

if len(sys.argv) < 2:
    print('Usage: python test.py "path_to_image.jpg"')
    raise SystemExit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
image_size = tuple(saved["image_size"])

image = cv2.imread(str(image_path))
if image is None:
    raise ValueError("Could not read the image.")

image = cv2.resize(image, image_size)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hog = cv2.HOGDescriptor(
    (128, 128), (16, 16), (8, 8), (8, 8), 9
)

features = hog.compute(gray).flatten().reshape(1, -1)

prediction = model.predict(features)[0]
print(f"Predicted item: {prediction}")

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(features)[0]
    print("Class scores:")
    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label}: {probability:.1%}")

