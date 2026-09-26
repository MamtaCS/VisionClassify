# VisionClassify

## Multi-Class Image Classification using Machine Learning

VisionClassify is an image classification project that classifies images into **10 different object classes** using OpenCV and machine learning algorithms.

The project covers the complete workflow from dataset preparation and image preprocessing to model training and performance evaluation.

---

## Dataset

The dataset contains images belonging to 10 different object classes:

1. Ball
2. Balloon
3. Black Pen
4. Book
5. Chair
6. Cup
7. Mobile
8. Pen
9. Pencil
10. Watch

The images are organized into separate folders according to their class.

```text
DataSet/
├── ball/
├── ballon/
├── blackPen/
├── book/
├── chair/
├── cup/
├── Mobile/
├── Pen/
├── pencil/
└── watch/
```

---

## Project Workflow

```text
Dataset Collection
        ↓
Data Annotation
        ↓
Image Preprocessing
        ↓
OpenCV
        ↓
Feature Extraction
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Classification
        ↓
Performance Evaluation
        ↓
Model Comparison
```

---

## Machine Learning Algorithms

The following classification algorithms are implemented:

- Logistic Regression
- Decision Tree
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Support Vector Classifier (SVC)

---

## Image Processing

OpenCV is used for image processing and preparation.

The main preprocessing steps include:

- Reading images
- Resizing images
- Converting images into numerical arrays
- Normalizing image data
- Preparing features for machine learning

---

## Training

The processed dataset is divided into training and testing data.

The training dataset is used to train the six machine learning models, while the testing dataset is used to evaluate their performance.

---

## Performance Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The performance of all six algorithms is compared to understand how each model performs on the image classification task.

---

## Project Structure

```text
VisionClassify/
│
├── DataSet/
│   ├── ball/
│   ├── ballon/
│   ├── blackPen/
│   ├── book/
│   ├── chair/
│   ├── cup/
│   ├── Mobile/
│   ├── Pen/
│   ├── pencil/
│   └── watch/
│
├── result/
│
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train.py
```

### 3. Test the models

```bash
python test.py
```

---

## Results

The results of the classification models will be stored in the `result/` folder.

The models will be compared using different performance metrics such as accuracy, precision, recall and F1-score.

---

## Future Improvements

- Increase the number of images per class
- Add data augmentation
- Improve feature extraction
- Tune model parameters
- Add real-time camera-based classification
- Compare traditional machine learning models with CNN models

---

## Conclusion

VisionClassify demonstrates the implementation of a multi-class image classification system using OpenCV and traditional machine learning algorithms.

The project provides a complete workflow covering **dataset preparation, preprocessing, model training, classification and performance evaluation**.
