# Signature-Classification-Using-CNN
![image](https://github.com/user-attachments/assets/0e5eb497-5d6a-488d-9905-3c5762457d26)

This project focuses on classifying student signatures using deep learning techniques, specifically a Convolutional Neural Network (CNN). The dataset contains signatures from 184 students, with 4 unique signatures for each student. The task is to correctly identify and classify each signature according to the corresponding student ID present in the input images.
Table of Contents

    Introduction
    Methodology
        Pre-Processing
        Model Architecture
    Results
    Discussion
    Instructions to Run
    Technologies Used

Introduction

The goal of this project is to develop a deep learning model capable of classifying the signatures of 184 students. For each student, 4 signatures are available, making the task multi-class classification with 184 classes.
The challenge lies in creating a robust CNN architecture that can generalize well, given the limited data, and accurately identify the signatures.
Methodology
Pre-Processing

    Reading and Scaling Images
        All images are resized to 888x1212 pixels.
        Each image is also scaled down by 50% for optimization.

    Extracting Signatures and IDs
        A function extracts all signatures from the provided images.
        Extracted signatures are saved in folders named after the corresponding student IDs.

    Data Augmentation
        Augmentation techniques are applied to increase the quantity of data and improve model generalization.

Model Architecture

The CNN architecture consists of several convolutional layers with Batch Normalization to improve gradient flow, followed by Max Pooling for dimensionality reduction. The model has the following layers:

    Block 1:
        2 Conv2D layers with 64 filters (3x3) and ReLU activation.
        Batch Normalization + Max Pooling.

    Block 2:
        2 Conv2D layers with 128 filters (3x3) and ReLU activation.
        Batch Normalization + Max Pooling.

    Block 3:
        2 Conv2D layers with 256 filters (3x3) and ReLU activation.
        Batch Normalization + Max Pooling.

    Block 4:
        2 Conv2D layers with 512 filters (3x3) and ReLU activation.
        Batch Normalization + Max Pooling.

    GlobalAveragePooling2D Layer:
        Replaces the Flatten layer to reduce overfitting.

    Fully Connected Layers:
        Dense layer with 1024 neurons.
        Dense layer with 512 neurons.
        Final Dense layer with 184 neurons (equal to the number of classes) and softmax activation for multi-class classification.

Results

The CNN achieved the following performance metrics:

    Train Accuracy: 94%
    Test Accuracy: 91.81%
    Train Loss: 0.22
    Precision: 93.35%
    Recall: 91.81%
    F1 Score: 91.68%


Model Performance
Discussion

    Data Augmentation: Due to the limited dataset, data augmentation was necessary to enhance generalization.
    High Complexity: With 184 classes, the model required a complex architecture to capture the relationships effectively.
    Training Challenges:
        The model occasionally drifted away from the optimal solution after reaching a minimum loss.
        To avoid overfitting, training was stopped when validation accuracy crossed 90%.


Technologies Used

    Python (Programming Language)
    TensorFlow / Keras (Deep Learning Framework)
    OpenCV (Image Processing Library)
    NumPy (Numerical Computation)
    Matplotlib (Visualization)
