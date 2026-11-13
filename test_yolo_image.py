
import cv2
import numpy as np
import onnxruntime as ort

# Class names (same order as training)
CLASSES = [
    "airpods", "apple", "bottle", "glasses", "mobile",
    "notebook", "pen", "perfume", "remote", "watch"
]

MODEL_PATH = "models/best.onnx"
IMAGE_PATH = "test_images/glasses_new.jpg"

# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# Read image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

original_h, original_w = image.shape[:2]

# Resize and preprocess
resized = cv2.resize(image, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

input_tensor = rgb.astype(np.float32) / 255.0
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)

# Run detection
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor})[0]

# YOLO output: [1, 14, 8400]
predictions = output[0].T

boxes = []
scores = []
class_ids = []

scale_x = original_w / 640
scale_y = original_h / 640

for prediction in predictions:
    class_scores = prediction[4:]
    class_id = int(np.argmax(class_scores))
    confidence = float(class_scores[class_id])

    if confidence < 0.25:
        continue

    cx, cy, w, h = prediction[:4]

    x1 = int((cx - w / 2) * scale_x)
    y1 = int((cy - h / 2) * scale_y)
    box_w = int(w * scale_x)
    box_h = int(h * scale_y)

    boxes.append([x1, y1, box_w, box_h])
    scores.append(confidence)
    class_ids.append(class_id)

# Remove duplicate overlapping boxes
indices = cv2.dnn.NMSBoxes(
    boxes, scores, 0.25, 0.45
)

if len(indices) > 0:
    for i in np.array(indices).flatten():
        x, y, w, h = boxes[i]
        label = CLASSES[class_ids[i]]
        confidence = scores[i]

        cv2.rectangle(
            image, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        text = f"{label}: {confidence:.2f}"

        cv2.putText(
            image, text, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        print(f"Detected: {label} | Confidence: {confidence:.2%}")
else:
    print("No object detected.")

# Save result
cv2.imwrite("detection_result.jpg", image)
print("Result saved as detection_result.jpg")
