# 🐾 Dog Breed Classifier

# Authors:

Aaron Nguyen @nguyen-aaron

# Description:

This project explores the use of convolutional neural networks (CNNs) to identify various dog breeds from images. Given an image of a dog, our model aims to accurately predict its breed from a labeled dataset of examples. Dog breed classification presents an interesting computer vision challenge, as many breeds share similar physical characteristics such as shape, size, and color. We will use the Stanford Dogs Dataset to train and evaluate our model’s performance. Our goal is to demonstrate how CNNs can capture spatial patterns in textures and edges to distinguish between visually similar classes in complex image data.

# Project Outline / Plan:

## Data Collection

### Data Preprocessing (Aaron):

Load and organize the Stanford Dogs dataset.

Split into training, validation, and test sets.

Apply data augmentation (rotation, zoom, flip) to improve generalization.

### Data Preprocessing (Alex):

Use the dataset from https://github.com/AtharvaTaras/Dog-Breeds-Dataset. 

Split into training and test sets. 

Apply data augmentation to the test dataset.

## Model Plans

### Model Construction (Aaron):

Build a convolutional neural network (CNN) using PyTorch.

Use the Stanford Dogs Dataset for training and evaluation.

Start with a base model (2 convolutional layers), then increase complexity by adding more layers and filters.

Loss: CrossEntropyLoss

Optimizer: Adam

Evaluate performance with: Accuracy and loss curves, confusion matrix, ROC and AUC curves

### Model Construction (Alex)

Build a CNN using PyTorch.

Start with 2 convolution layers and 2 fully connected layers.

Use Cross Entropy Loss and the Adam optimizer.

## Project Timeline
