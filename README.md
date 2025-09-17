# Diabetic Retinopathy Detection

This project implements a deep learning model to detect diabetic retinopathy (DR) from retinal fundus images and classify its severity. The model is built using PyTorch and leverages a pre-trained `EfficientNet-B0` from the `timm` library. 

This solution is based on the problem presented in the **APTOS 2019 Blindness Detection** competition on Kaggle.

## Project Overview

Diabetic Retinopathy is the leading cause of blindness among working-age adults. Early detection and treatment can prevent vision loss. This project automates the detection process by classifying fundus images into five stages of DR:

| Class | Diagnosis                  |
| :---- | :------------------------- |
| 0     | No DR                      |
| 1     | Mild                       |
| 2     | Moderate                   |
| 3     | Severe                     |
| 4     | Proliferative DR           |

The model achieves a **Quadratic Weighted Cohen's Kappa score of 0.8282** on the validation set after 5 epochs of training.

![Sample Retinal Fundus Images](/images/samples.png)

## Methodology

### 1. Exploratory Data Analysis (EDA)

The dataset exhibits a significant class imbalance, with a majority of images belonging to the "No DR" (Class 0) category. This imbalance is a key challenge that needs to be addressed during training.

**Class Distribution:**<br>
![Diabetic Retinopathy Class Distribution Chart](/images/countplot.png)

To handle this, a **stratified train-validation split** was used to ensure that the class proportions in the training and validation sets were the same as in the original dataset.

### 2. Data Preprocessing & Augmentation

To make the model robust and prevent overfitting, extensive image augmentation was applied to the training data using the `albumentations` library.

* **Training Augmentations:**
    * Resize to 224x224
    * Horizontal & Vertical Flips
    * Random Brightness & Contrast Adjustments
    * Shift, Scale, & Rotate
    * Normalization

* **Validation/Test Preprocessing:**
    * Resize to 224x224
    * Normalization only

A custom PyTorch `Dataset` class (`DRDataset`) was created to efficiently load and transform the images.

### 3. Model Architecture

A pre-trained **EfficientNet-B0** model from the `timm` library was used as the backbone. The final classifier layer was replaced with a new one with 5 output units, corresponding to the five DR classes. 

### 4. Training & Evaluation

* **Framework:** PyTorch
* **Optimizer:** `AdamW` with a learning rate of `1e-4`.
* **Loss Function:** `CrossEntropyLoss`, suitable for multi-class classification.
* **Evaluation Metric:** The **Quadratic Weighted Cohen's Kappa** score was used for evaluation. This metric is ideal for ordinal classification problems (where classes have a natural order) and is sensitive to class imbalance. The model with the highest validation Kappa score was saved as the best model.

The model was trained for 5 epochs for this demonstration, achieving a strong Kappa score. Further improvements can be expected with more epochs.

## Results & Performance

After 5 epochs, the best model achieved the following performance on the validation set:

* **Best Validation Kappa Score:** **0.8282**
* **Final Validation Loss:** 0.6873

### Confusion Matrix

The confusion matrix shows that the model performs very well on Class 0 (No DR) and Class 2 (Moderate). It shows some confusion between adjacent classes, which is expected. The performance on minority classes (1, 3, and 4) is lower, a direct consequence of the data imbalance.

![Final model's confusion matrix](/images/confusion.png)

## Setup and Usage

Follow these steps to set up the environment and run the project.

### 1. Prerequisites

* Python 3.x
* PyTorch
* Kaggle account and API token (for downloading the dataset)

### 2. Installation

Clone the repository:
```
git clone https://github.com/notCliche/diabetic-retinopathy-classifier.git
cd diabetic-retinopathy-detector
```

Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3\. Dataset

Download the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset from Kaggle. Place the data in a directory structure as expected by the notebook. For example:

```
./input/aptos2019-blindness-detection/
├── train_images/
├── test_images/
├── train.csv
└── test.csv
```

## Authors
- Adarsh Dhakar
- Om Prakash Behera