# 🐶🐱 Dogs vs Cats Image Classification using CNN

## 📌 Project Overview

This project builds a Convolutional Neural Network (CNN) to classify images of dogs and cats.
It includes data preprocessing, model training, performance evaluation, and prediction on custom images.

---

## 🚀 Features

* End-to-end deep learning pipeline
* Image preprocessing using TensorFlow
* CNN model built from scratch
* Improved model with Batch Normalization & Dropout
* Performance visualization (accuracy & loss)
* Custom image prediction

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV
* Kaggle API

---

## 📂 Dataset

Dataset used: **Dogs vs Cats** from Kaggle
Link: https://www.kaggle.com/datasets/salader/dogsvscats

---

## ⚙️ Project Workflow

### 1. Data Loading

* Dataset downloaded using Kaggle API
* Extracted and loaded using `image_dataset_from_directory`

### 2. Preprocessing

* Image resizing (256x256)
* Normalization (pixel values scaled to [0,1])

### 3. Model Building

* CNN with Conv2D, MaxPooling layers
* Activation: ReLU
* Output: Sigmoid (Binary Classification)

### 4. Model Improvement

* Added:

  * Batch Normalization
  * Dropout (to reduce overfitting)

### 5. Training

* Loss: Binary Crossentropy
* Optimizer: Adam
* Metric: Accuracy

### 6. Evaluation

* Training vs Validation Accuracy
* Training vs Validation Loss

### 7. Prediction

* Custom images tested using OpenCV
* Output: Dog or Cat

---

## 📊 Results

* Model achieves good accuracy on validation data
* Improved model reduces overfitting compared to baseline CNN

---



## 📌 Future Improvements

* Use Transfer Learning (VGG16 / ResNet)
* Hyperparameter tuning
* Deploy using Streamlit
* Add confusion matrix & classification report

---

## 👤 Author

Amarpal Singh Dutta
(Data Science & Machine Learning Enthusiast)
