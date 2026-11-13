import onnxruntime as ort
import os

model_path = os.path.join("models", "best.onnx")

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("YOLO model loaded successfully!")
print("Input:", session.get_inputs()[0].shape)
print("Output:", session.get_outputs()[0].shape)