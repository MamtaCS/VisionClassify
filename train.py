
from pathlib import Path
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "DataSet"
RESULT_DIR = BASE_DIR / "result"
IMAGE_SIZE = (128, 128)

hog = cv2.HOGDescriptor(
    (128, 128), (16, 16), (8, 8), (8, 8), 9
)

def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).flatten()

X, y = [], []

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    count = 0
    for path in class_dir.iterdir():
        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            continue

        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        X.append(extract_features(image))
        y.append(class_dir.name)
        count += 1

    print(f"{class_dir.name}: {count} images loaded")

if len(set(y)) < 2:
    raise ValueError("Need at least two classes with readable images.")

X = np.array(X)
y = np.array(y)

print(f"\nTotal images: {len(y)}")
print("Classes:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", probability=True, class_weight="balanced")
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nTest accuracy:", f"{accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))

RESULT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {"model": model, "image_size": IMAGE_SIZE, "name": "HOG + SVM"},
    RESULT_DIR / "best_model.joblib"
)

print("Saved HOG model to:", RESULT_DIR / "best_model.joblib")
