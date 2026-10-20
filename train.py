from pathlib import Path
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "DataSet"
RESULT_DIR = BASE_DIR / "result"
IMAGE_SIZE = (128, 128)

hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)

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
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
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

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\nTotal images: {len(y)}")
print("Classes:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "K-Nearest Neighbors (KNN)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Classifier": make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
}

print("\n--- Model Training & Comparison ---")
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {"accuracy": acc, "model": model, "predictions": preds}
    print(f"{name:26s} Accuracy: {acc:.2%}")

best_name = max(results, key=lambda k: results[k]["accuracy"])
best_info = results[best_name]

print(f"\nSelected Best Model: {best_name} ({best_info['accuracy']:.2%})")

print("\n--- Classification Report ---")
print(classification_report(y_test, best_info["predictions"], zero_division=0))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, best_info["predictions"], labels=sorted(set(y)))
print(f"Labels: {sorted(set(y))}\n")
print(cm)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": best_info["model"], "image_size": IMAGE_SIZE, "name": best_name},
    RESULT_DIR / "best_model.joblib"
)
print(f"\nSaved best model ({best_name}) to: {RESULT_DIR / 'best_model.joblib'}")
