"""
VisionClassify - Real-Time Webcam Object Recognition
Classifies full webcam frames using the same HOG preprocessing as training.
"""

from pathlib import Path
import sys

import cv2
import joblib

from vision_utils import IMAGE_SIZE, classify_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "result" / "best_model.joblib"
DEFAULT_SCORE_THRESHOLD = 0.40

if not MODEL_PATH.is_file():
    alt_path = BASE_DIR / "VisionClassify" / "result" / "best_model.joblib"
    if alt_path.is_file():
        MODEL_PATH = alt_path

if not MODEL_PATH.is_file():
    print(f"\n[ERROR] Model file not found at: {MODEL_PATH}")
    print("Please train the model first by running:")
    print("    python train.py\n")
    sys.exit(1)

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    image_size = tuple(saved.get("image_size", IMAGE_SIZE))
    model_name = saved.get("name", "Classifier")
    score_threshold = float(
        saved.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
    )
except (OSError, KeyError, TypeError, ValueError) as err:
    print(f"\n[ERROR] Could not load the trained model: {err}")
    sys.exit(1)

camera = cv2.VideoCapture(0)
success, first_frame = camera.read() if camera.isOpened() else (False, None)
if not success or first_frame is None:
    camera.release()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    success, first_frame = camera.read() if camera.isOpened() else (False, None)

if not success or first_frame is None:
    print("\n[ERROR] Could not access the webcam.")
    print("Check camera permissions and ensure no other app is using it.\n")
    camera.release()
    sys.exit(1)

print("\n" + "=" * 50)
print(" VisionClassify: Real-Time Object Recognition")
print("=" * 50)
print(" Keep one object clearly visible near the center.")
print(" Press 'Q' or Escape to quit.")
print("=" * 50 + "\n")

try:
    while True:
        if first_frame is not None:
            frame = first_frame
            first_frame = None
            success = True
        else:
            success, frame = camera.read()
        if not success or frame is None:
            print("[WARNING] Could not read frame from webcam. Exiting...")
            break

        height, width = frame.shape[:2]
        prediction, candidate, score, _ = classify_image(
            model,
            frame,
            score_threshold,
            image_size,
        )

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 78), (25, 25, 25), -1)
        cv2.rectangle(overlay, (0, height - 35), (width, height), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        box_size = int(min(height, width) * 0.6)
        x1 = (width - box_size) // 2
        y1 = (height - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
        cv2.putText(
            frame,
            "Keep object near center",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

        if prediction is None:
            result_text = f"Unknown Object (top: {candidate})"
            color = (0, 180, 255)
        else:
            result_text = f"Class: {prediction.capitalize()}"
            color = (0, 255, 128)
        if score is not None:
            result_text += f"  Score: {score:.1%}"

        cv2.putText(
            frame,
            result_text,
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Model: {model_name} | Unknown below {score_threshold:.0%}",
            (16, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press 'Q' or Esc to quit",
            (15, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("VisionClassify - Real-Time Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("[INFO] Quit requested by user.")
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released and windows closed. Goodbye!")
