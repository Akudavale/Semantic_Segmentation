# Semantic Segmentation with DeepLabV3

This repository contains code for training and evaluating a semantic segmentation model using the DeepLabV3 architecture with a ResNet-50 backbone on the Cityscapes dataset. The implementation leverages PyTorch for model training and evaluation.

## Overview

The code provided sets up a semantic segmentation pipeline using DeepLabV3. Key components include:

- **CityscapesDataset**: A custom PyTorch Dataset for loading and preprocessing images and labels from the Cityscapes dataset.
- **Model Configuration**: A DeepLabV3 model pre-trained on ImageNet, adapted for the Cityscapes dataset with 34 classes.
- **Training and Evaluation**: Functions for training the model, validating it, and computing metrics such as IoU and pixel accuracy.

## Requirements

The following Python packages are required:

- PyTorch
- torchvision
- numpy
- OpenCV
- scikit-learn

Install the required packages using pip:

```bash
pip install torch torchvision numpy opencv-python scikit-learn
```

## Dataset
You need the Cityscapes dataset, which can be downloaded from Cityscapes dataset website. Place the dataset in a directory structure like:

```bash
Datasets/
  ├── leftImg8bit/
  │   └── train/
  │   └── val/
  │   └── test/
  └── gtFine/
      └── train/
      └── val/
      └── test/
 ```

## Code Explanation

### Dataset Class
CityscapesDataset is a PyTorch Dataset class that handles loading and transforming images and their corresponding labels.

### DataLoader
DataLoaders are created for training, validation, and test datasets. Batches of data are shuffled for training and loaded in a non-shuffled manner for validation and testing.

### Model Configuration
A pre-trained DeepLabV3 model is modified to output predictions for 34 classes. The model is moved to the appropriate device (GPU if available).

### Training and Evaluation
Training: The train function handles one epoch of training. It computes the loss using CrossEntropyLoss and updates the model parameters.
Validation: The validate function evaluates the model on the validation set without updating the model parameters.
Metrics: The compute_metrics function calculates the Intersection over Union (IoU) and pixel accuracy for model evaluation.

### Usage
Run the script to train and evaluate the model:

```bash
python Semantic_Segmentation.py
```
### Training
The training process runs for a predefined number of epochs. Loss values for both training and validation are printed out.

### Evaluation
After training, the script computes and prints the mean IoU and pixel accuracy for both validation and test datasets.

## Acknowledgements
### DeepLabV3:
The model is based on the DeepLabV3 architecture as described in DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation.
### Cityscapes Dataset: 
Used with permission from the Cityscapes dataset creators.
