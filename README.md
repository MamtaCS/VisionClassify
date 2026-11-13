# VisionClassify – Real-Time Object Detection

VisionClassify is a Python-based real-time object detection system that uses **YOLO11n** to identify everyday objects through images and a live webcam.

The project uses a custom-trained YOLO model with 10 object categories. It detects objects, draws bounding boxes around them, and displays their predicted class names and confidence scores.

The trained model is exported to ONNX format and runs locally using ONNX Runtime, allowing object detection without requiring PyTorch on the target Windows machine.

## Problem Statement and Objectives

Recognizing everyday objects automatically is a useful application of computer vision. However, objects can appear different depending on lighting, background, camera angle, distance and environment.

The objectives of VisionClassify are to:

- Develop a custom object detection model for everyday objects.
- Detect multiple objects and locate them using bounding boxes.
- Perform real-time detection through a webcam.
- Display predicted object categories and confidence scores.
- Run the trained model locally using ONNX Runtime.
- Build a simple and practical computer vision application.

## Supported Object Categories

VisionClassify currently detects the following 10 categories:

1. AirPods
2. Apple
3. Bottle
4. Glasses
5. Mobile
6. Notebook
7. Pen
8. Perfume
9. Remote
10. Watch

## Technologies Used

- **Python** – Main programming language.
- **YOLO11n** – Object detection model.
- **Ultralytics** – Model training and ONNX export.
- **OpenCV** – Image processing, webcam access and bounding-box visualization.
- **NumPy** – Numerical operations and image preprocessing.
- **ONNX Runtime** – Local execution of the exported model.
- **Google Colab** – Model training using an NVIDIA GPU.

## Project Structure

```text
VisionClassify/
│
├── DataSet/                  # Original object images
│
├── models/
│   └── best.onnx             # Trained YOLO model
│
├── test_images/              # Images for testing
│
├── YOLO_Dataset/             # Training and validation dataset
│
├── train.py                  # Previous Random Forest training script
├── test.py                   # Previous image classification script
├── live_detect.py            # Previous webcam classification script
│
├── test_yolo.py              # ONNX model loading test
├── test_yolo_image.py        # YOLO image detection
├── webcam_yolo.py            # YOLO live webcam detection
│
├── capture_dataset.py        # Dataset image collection
├── vision_utils.py           # Previous HOG utilities
│
├── requirements.txt
├── README.md
└── .gitignore
```

The previous HOG + Random Forest scripts are retained as part of the project's development history. The current YOLO detection workflow uses `models/best.onnx`, `test_yolo_image.py` and `webcam_yolo.py`.

## Dataset and Annotation

A custom dataset was prepared for 10 object categories.

The annotated YOLO dataset contains:

| Dataset split | Images |
|---|---:|
| Training | 140 |
| Validation | 30 |
| **Total** | **170** |

Each image has a corresponding YOLO-format annotation file containing the object class ID and normalized bounding-box coordinates.

The dataset was divided into training and validation sets with all 10 categories represented.

Each class has 17 annotated images in the combined dataset.

## Model Training

The project uses the lightweight **YOLO11n (Nano)** architecture.

Training was performed in Google Colab using an NVIDIA Tesla T4 GPU.

Training configuration:

- Model: YOLO11n
- Image size: 640 × 640
- Maximum epochs: 100
- Batch size: 16
- Early stopping patience: 20
- Training stopped at epoch 46.
- Best model saved from epoch 26.

Early stopping automatically stopped training when validation performance did not improve.

## Model Evaluation

The best model was evaluated on the 30-image validation set.

| Metric | Result |
|---|---:|
| Precision | 86.7% |
| Recall | 82.4% |
| mAP@50 | 88.6% |
| mAP@50–95 | 60.2% |

These metrics are based on a small validation dataset and should be treated as preliminary results rather than a guarantee of real-world accuracy.

### Class-wise mAP@50

| Object | mAP@50 |
|---|---:|
| AirPods | 99.5% |
| Apple | 99.5% |
| Bottle | 83.0% |
| Glasses | 99.5% |
| Mobile | 91.3% |
| Notebook | 99.5% |
| Pen | 66.5% |
| Perfume | 75.3% |
| Remote | 99.5% |
| Watch | 72.0% |

## ONNX Model Export

After training, the best YOLO model was exported into ONNX format.

The exported model is stored at:

```text
models/best.onnx
```

ONNX Runtime allows the model to run locally without requiring the PyTorch framework.

Model input shape:

```text
[1, 3, 640, 640]
```

Model output shape:

```text
[1, 14, 8400]
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd VisionClassify
```

### 2. Create a virtual environment

```bash
python -m venv .venv-yolo
```

Activate it on Windows:

```powershell
.venv-yolo\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install opencv-python numpy onnxruntime
```

## Running the Project

### 1. Verify ONNX Model Loading

```bash
python test_yolo.py
```

This checks whether the trained ONNX model loads successfully and displays its input and output shapes.

### 2. Detect Objects in an Image

Place a test image inside the `test_images` folder.

For example:

```text
test_images/
└── glasses.jpg
```

Run:

```bash
python test_yolo_image.py
```

The script predicts the object category, draws a bounding box and saves the result as:

```text
detection_result.jpg
```

### 3. Real-Time Webcam Detection

Run:

```bash
python webcam_yolo.py
```

The application opens the webcam and performs live object detection.

It displays:

- Detected object name
- Confidence score
- Bounding box around the detected object

Press **Q** to close the webcam window.

## Features

- Custom-trained YOLO11n object detection.
- Detection of 10 everyday object categories.
- Bounding-box visualization.
- Confidence-score display.
- Still-image object detection.
- Real-time webcam detection.
- Local execution using ONNX Runtime.
- No database, web server or external API required.

## Limitations

- The annotated training dataset contains only 170 images.
- Validation results are based on a small sample of 30 images.
- Detection performance may vary with lighting, camera quality, distance, background and object orientation.
- Objects outside the 10 trained categories are not reliably recognized.
- Confidence scores do not guarantee prediction correctness.
- CPU-based webcam inference performance depends