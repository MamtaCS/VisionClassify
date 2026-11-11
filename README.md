# VisionClassify

VisionClassify is a beginner-friendly, local Python project for classifying
images of ten everyday objects. It uses OpenCV to preprocess images and
extract Histogram of Oriented Gradients (HOG) features, then uses a
scikit-learn Random Forest classifier to predict an object's category.

The project supports still-image prediction and live webcam classification.
It predicts one class for an image or full webcam frame; it is not an object
detector and does not locate multiple objects.

## Problem statement and objectives

Images of the same object can look different when the background, lighting,
angle, distance, or camera changes. This project demonstrates a complete
image-classification workflow for a small set of everyday objects.

Its objectives are to:

- Organize example images by category and use folder names as labels.
- Apply consistent image preprocessing and HOG feature extraction.
- Compare Random Forest sizes using grouped validation.
- Evaluate predictions on a held-out grouped test split.
- Predict a category from a still image or webcam frame.
- Return `Unknown Object` for predictions whose model score is below a
  validation-selected threshold.

## Technologies

- **Python**
- **OpenCV** (`opencv-python`) for image loading, preprocessing, HOG features,
  webcam capture, and display
- **NumPy** for numerical operations
- **scikit-learn** for the Random Forest, grouped cross-validation, and
  evaluation metrics
- **Joblib** for saving and loading the trained model

The application runs locally. It does not require a database, web server, or
external API.

## Supported object categories

The project currently uses these ten dataset folders as class labels:

1. `airpods`
2. `apple`
3. `bottle`
4. `glasses`
5. `mobile`
6. `notebook`
7. `pen`
8. `perfume`
9. `remote`
10. `watch`

`train.py` expects ten categories. Put each image inside the folder named for
its category.

## Project structure

```text
VisionClassify/
├── DataSet/
│   ├── airpods/
│   ├── apple/
│   ├── bottle/
│   ├── glasses/
│   ├── mobile/
│   ├── notebook/
│   ├── pen/
│   ├── perfume/
│   ├── remote/
│   └── watch/
├── result/
│   └── best_model.joblib       # Generated model; result/ is Git-ignored
├── capture_dataset.py          # Capture labeled images from a webcam
├── live_detect.py              # Live webcam classification
├── test.py                     # Still-image classification
├── train.py                    # Grouped model selection, evaluation, and training
├── vision_utils.py             # Shared preprocessing and prediction logic
├── requirements.txt            # Python dependencies
└── README.md
```

The example images `glasses_new.jpg` and `pen_new.jpg`, when present at the
project root, are not automatically included in training. Pass an image path
explicitly to `test.py` to classify it.

## Dataset

The current dataset contains **551 images**:

| Category | Images |
|---|---:|
| airpods | 66 |
| apple | 50 |
| bottle | 51 |
| glasses | 60 |
| mobile | 51 |
| notebook | 52 |
| pen | 50 |
| perfume | 60 |
| remote | 52 |
| watch | 59 |
| **Total** | **551** |

The dataset was checked during the latest training run: all 551 images were
readable. Counts are reasonably balanced, but some files are very similar
captures of the same scene. Grouped splitting keeps detected near-duplicate
frames together rather than placing similar images in both training and test
data. The current images also have limited capture diversity, so test results
should not be assumed to represent every camera, background, or environment.

## Image preprocessing and HOG

Training, still-image prediction, and webcam prediction share the feature
extraction function in `vision_utils.py`:

1. If bright-green capture text is detected in the upper-left region, that
   region is cleaned in memory using OpenCV inpainting. Dataset files are not
   modified. This detection targets the capture script's green preview label.
2. The image is resized to **128 × 128** pixels.
3. The resized image is converted from BGR to grayscale.
4. An OpenCV HOG descriptor extracts edge and gradient features. Its window is
   128 × 128, with 16 × 16 blocks, 8 × 8 block strides, 8 × 8 cells, and 9
   orientation bins.

The same HOG preprocessing is used for training, still images, and full webcam
frames. Resizing to a square can distort an image's aspect ratio. Inpainting
is a targeted cleanup step, not a general image-enhancement method; green
content in the targeted area could also be affected.

## Model training and evaluation

`train.py` extracts HOG features from the labeled images and groups same-class
near-duplicates using a perceptual hash. It then:

1. Reserves a grouped fold for held-out testing using five-fold
   `StratifiedGroupKFold` splitting.
2. Compares Random Forest models with 100 and 300 trees on the remaining data
   using four-fold grouped cross-validation.
3. Selects the tree count with the higher mean macro-F1.
4. Prints grouped cross-validation results, held-out accuracy, per-class
   precision/recall/F1, a confusion matrix, and Unknown prediction coverage.
5. Retrains the selected Random Forest on all available images and saves it to
   `result/best_model.joblib`.

### Latest grouped test results

The latest run selected a **300-tree Random Forest**. On the separate
held-out grouped test fold (**111 images**), it achieved:

- **Accuracy:** 93.69%
- **Macro-F1:** 92.98%

At the selected **30%** score threshold, **76.58%** of test images were
accepted as known classes; accepted predictions were **100% accurate on this
test fold**, and 26 images were returned as `Unknown Object`.

Notebook recognition is currently weak. In the latest test fold, notebook
recall was **55%** (6 of 11 notebook images correctly classified). The
notebook errors were predicted as apple (2), airpods (1), bottle (1), and
remote (1). The small test sample means these class metrics can vary when the
dataset or split changes.

## Unknown Object and score threshold

The saved model contains a **30% threshold**, selected from grouped
out-of-fold validation predictions in the latest run. When the top predicted
class's Random Forest vote fraction is below this threshold, `test.py` and
`live_detect.py` display `Unknown Object` rather than the top class.

This threshold can reduce forced low-scoring predictions; it **does not
guarantee perfect recognition** or reliably detect every unfamiliar object.
A Random Forest vote fraction is not a calibrated probability. Unknown
objects can still receive a score above the threshold, and valid class images
can be rejected as Unknown.

## Installation and usage

Run commands from the project directory.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Capture more training images (optional)

For example, to add images to the `mobile` category:

```bash
python capture_dataset.py mobile
```

Press **Space** to save an image and **Q** to exit. The preview shows a green
category/count label, but the saved image does not include that text. Use only
the ten supported category names so the training dataset still has the
expected classes. Capture multiple examples with varied physical objects,
backgrounds, lighting, angles, and distances.

### 3. Train and evaluate

```bash
python train.py
```

Training prints validation and held-out test results, then saves the updated
model to `result/best_model.joblib`. The `result/` directory is ignored by
Git; run training in a fresh checkout before making predictions if the model
file is not present.

### 4. Classify a still image

```bash
python test.py "path_to_image.jpg"
```

The script prints the model, predicted category or `Unknown Object`, and
Random Forest vote fractions for each class.

### 5. Run live webcam classification

```bash
python live_detect.py
```

Keep one object clearly visible near the center of the camera view. Press
**Q** or **Esc** in the webcam window to quit. The script first tries the
default camera backend and falls back to DirectShow if the first backend
cannot read a frame.

## Limitations and future improvements

- **Notebook recognition is weak**, as shown by its 55% recall on the latest
  held-out test fold.
- The dataset is small and has limited variety in backgrounds, lighting,
  distance, and camera conditions. Additional clean real-world captures are
  the most important next step.
- The system predicts one label for a whole image or webcam frame. It does not
  locate objects, handle multiple objects, or identify objects outside the
  ten trained categories reliably.
- The square resize can distort image shape, and grayscale HOG does not
  represent color.
- The Unknown threshold is based on validation scores; it is not a guarantee
  of open-set recognition.
- In the latest run, in-memory overlay cleanup removed a class-text shortcut,
  but targeted inpainting can affect other bright-green pixels near the
  upper-left corner.
- Future improvements could include collecting more varied images, reviewing
  labels and capture quality, evaluating on images from a separate camera or
  environment, and testing more robust preprocessing or feature/model
  approaches against grouped validation before adopting them.
