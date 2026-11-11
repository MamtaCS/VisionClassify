from pathlib import Path
import sys

import cv2
import joblib

from vision_utils import IMAGE_SIZE, classify_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "result" / "best_model.joblib"
DEFAULT_SCORE_THRESHOLD = 0.40

if len(sys.argv) < 2:
    print('Usage: python test.py "path_to_image.jpg"')
    raise SystemExit(1)

image_path = Path(sys.argv[1])
if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. Run train.py first."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
image_size = tuple(saved.get("image_size", IMAGE_SIZE))
model_name = saved.get("name", "Model")
score_threshold = float(
    saved.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
)

image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Could not read the image: {image_path}")

prediction, candidate, score, probabilities = classify_image(
    model,
    image,
    score_threshold,
    image_size,
)
print(f"Model: {model_name}")
if prediction is None:
    print(f"Predicted item: Unknown Object (top class: {candidate})")
else:
    print(f"Predicted item: {prediction}")
if score is not None:
    print(f"Model score: {score:.1%} (Unknown threshold: {score_threshold:.0%})")
if probabilities is not None:
    print("Class scores (Random Forest vote fractions, not calibrated probabilities):")
    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label}: {probability:.1%}")
