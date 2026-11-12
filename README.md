# Analysis-of-a-Real-World-Dataset
# Digits That Matter: Binary Classification of Digit 3

## Project Overview

This project builds a binary machine learning classifier using the **MNIST handwritten-digit dataset**. The goal is to identify whether an image contains the handwritten digit **3** or any other digit from 0 to 9.

The classifier uses a **Stochastic Gradient Descent (SGD) Classifier** from Scikit-learn. The project includes data preparation, cross-validation, model training, test-set evaluation, confusion matrix analysis, ROC curve analysis, and decision-threshold tuning.

**Classification Labels:**
- `True` – The image represents the digit 3.
- `False` – The image represents any digit other than 3.

## Project Objectives

- Load and prepare the MNIST dataset.
- Convert the original ten-class digit labels into a binary target.
- Normalize image pixel values.
- Train an SGD binary classifier.
- Evaluate performance using 3-fold stratified cross-validation.
- Test the trained model on a separate held-out test set.
- Analyze precision, recall, F1 score, accuracy, and the confusion matrix.
- Evaluate the classifier using an ROC curve and ROC-AUC.
- Explore how changing the decision threshold affects classification performance.

## Dataset

The project uses the **MNIST dataset**, loaded from OpenML with Scikit-learn's `fetch_openml()` function.

| Dataset Detail | Description |
|---|---|
| Dataset | MNIST (`mnist_784`) |
| Total Images | 70,000 |
| Image Dimensions | 28 × 28 pixels |
| Features per Image | 784 pixel values |
| Training Images | 60,000 |
| Test Images | 10,000 |
| Target | Digit 3 versus all other digits |

The first 60,000 images are used for training and the final 10,000 images are reserved for testing. Pixel values are converted to the range 0–1 by dividing by 255.

## Technologies and Libraries

- **Python** – Programming language
- **Jupyter Notebook** – Development environment
- **NumPy** – Numerical operations and image data preparation
- **Pandas** – Data organization and metric tables
- **Matplotlib** – Visualizations and ROC curve
- **Scikit-learn** – Dataset loading, classification, cross-validation, and evaluation

## Machine Learning Workflow

### 1. Load and Prepare the Data

The MNIST dataset is downloaded from OpenML. The image data and digit labels are separated into training and testing sets.

The original labels are converted into binary labels: digit 3 is `True`, and every other digit is `False`.

### 2. Normalize Pixel Values

The original pixel values range from 0 to 255. They are divided by 255 so that all pixel values fall between 0 and 1.

### 3. Cross-Validation

A **3-fold StratifiedKFold** cross-validation strategy is used on the training data. The `cross_val_predict()` function generates out-of-fold predictions so that each training image is predicted by a model that did not train on that image.

### 4. Train the Classifier

An `SGDClassifier` is trained using all 60,000 training images. The random state is set to 42 for reproducibility.

### 5. Evaluate the Model

The trained model predicts labels for the 10,000 held-out test images. The notebook evaluates accuracy, precision, recall, F1 score, and the confusion matrix.

### 6. ROC Curve and AUC

The model's continuous decision scores are used to calculate the ROC curve and ROC-AUC. This evaluates how well the classifier separates digit 3 from all other digits across possible decision thresholds.

### 7. Error Analysis

The notebook identifies:
- False positives: Other digits incorrectly predicted as 3.
- False negatives: Actual 3s incorrectly predicted as not 3.

It also counts which other digit labels appear most often among false positives.

### 8. Decision Threshold Tuning

The notebook tests decision thresholds of -1.0, -0.5, 0.0, 0.5, and 1.0. This demonstrates the trade-off between precision and recall when changing the classifier's decision boundary.

## Results

### Held-Out Test-Set Performance

| Metric | Result |
|---|---:|
| Accuracy | 0.9755 (97.55%) |
| Precision | 0.9682 (96.82%) |
| Recall | 0.7832 (78.32%) |
| F1 Score | 0.8659 |
| ROC-AUC | 0.9805 |

### Confusion Matrix

The test-set confusion matrix is:

| | Predicted: Not 3 | Predicted: 3 |
|---|---:|---:|
| **Actual: Not 3** | 8,964 | 26 |
| **Actual: 3** | 219 | 791 |

- **True Negatives (TN):** 8,964
- **False Positives (FP):** 26
- **False Negatives (FN):** 219
- **True Positives (TP):** 791

The model correctly identifies many images that are not digit 3 and makes relatively few false-positive predictions. However, it misses some actual digit 3 images, as reflected in its recall score.

### Cross-Validation Results

The 3-fold cross-validation results on the training set are:

| Metric | Result |
|---|---:|
| Accuracy | 0.9711 |
| Precision | 0.8879 |
| Recall | 0.8206 |
| F1 Score | 0.8529 |

### False-Positive Analysis

Among the non-3 digits incorrectly classified as 3, the most frequent labels are:

| Actual Digit | False-Positive Count |
|---|---:|
| 5 | 14 |
| 2 | 8 |
| 8 | 3 |
| 1 | 1 |

The digit 5 is the most frequent source of false-positive predictions in this test set.

### Decision Threshold Analysis

The notebook compares five decision thresholds.

At the default threshold of 0.0:
- Precision: 0.9682
- Recall: 0.7832

Lowering the threshold to -0.5 increases recall to 0.8515, while precision decreases to 0.9389.

Raising the threshold to 0.5 increases precision to 0.9835 but lowers recall to 0.7089.

This illustrates the trade-off between identifying more actual digit 3 images and avoiding false-positive predictions.

## How to Run the Project

1. Clone or download this repository.
2. Install Python and Jupyter Notebook.
3. Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `MNIST_Digit_3_Binary_Classifier(1).ipynb`.
6. Run the notebook cells from top to bottom.

**Important:** The notebook downloads MNIST from OpenML when the dataset-loading cell runs, so an internet connection is required the first time the dataset is loaded.

## Repository Files

| File | Description |
|---|---|
| `MNIST_Digit_3_Binary_Classifier(1).ipynb` | Jupyter Notebook containing data preparation, model training, evaluation, and threshold tuning |
| `README.md` | Project documentation |

## Key Learning Outcomes

- Understanding binary classification.
- Preparing image data for machine learning.
- Converting multiclass labels into binary labels.
- Normalizing image pixel values.
- Training an SGD classifier with Scikit-learn.
- Applying stratified cross-validation.
- Interpreting accuracy, precision, recall, and F1 score.
- Understanding true positives, true negatives, false positives, and false negatives.
- Evaluating classification performance with ROC curves and ROC-AUC.
- Understanding the relationship between decision thresholds, precision, and recall.

## Limitations and Future Improvements

This project provides a baseline binary classifier for recognizing digit 3. Future improvements could include comparing additional classification algorithms, tuning model hyperparameters, examining incorrectly classified images, and testing the model on handwritten digits from sources beyond MNIST.

## Real-World Applications

Handwritten digit recognition is used in several practical applications, including:

- Automated check processing in banking.
- Postal code recognition in mail sorting.
- Digit recognition in scanned forms.
- Automated document processing.

This project demonstrates the fundamentals of a technology used in these applications.

## Author

**Fenny Patel**

---

*This project was completed as part of a machine learning classification assignment.*
