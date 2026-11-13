
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = [
    "airpods", "apple", "bottle", "glasses", "mobile",
    "notebook", "pen", "perfume", "remote", "watch"
]

session = ort.InferenceSession(
    "models/best.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

print("VisionClassify Webcam Started!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    output = session.run(None, {input_name: tensor})[0]
    predictions = output[0].T

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < 0.25:
            continue

        cx, cy, bw, bh = pred[:4]

        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        box_w = int(bw * w / 640)
        box_h = int(bh * h / 640)

        boxes.append([x1, y1, box_w, box_h])
        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, 0.25, 0.45
    )

    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, bw, bh = boxes[i]

            label = CLASSES[class_ids[i]]
            confidence = scores[i]

            cv2.rectangle(
                frame, (x, y), (x + bw, y + bh),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"{label}: {confidence:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

    cv2.imshow("VisionClassify - Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
