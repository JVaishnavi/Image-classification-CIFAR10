# CIFAR-10 Classification: Custom CNN Architecture (S9 Assignment)

This repository contains a PyTorch implementation of a custom Convolutional Neural Network (CNN) designed to classify the CIFAR-10 dataset. The solution focuses on parameter efficiency, receptive field expansion, and advanced data augmentation using the Albumentations library.

## Key Objectives & Constraints

This project was built to satisfy the following strict requirements:
* **Architecture:** C1C2C3C4 structure (4 Convolutional Blocks).
* **No Max Pooling:** Downsampling is achieved via **Strided Convolutions** (3 layers with 3x3 kernel, stride=2).
* **Advanced Convolutions:** * Must include **Depthwise Separable Convolution**.
    * Must include **Dilated Convolution**.
* **Global Average Pooling (GAP):** Used at the end of the network (followed by optional FC layer).
* **Receptive Field:** Total RF > 44.
* **Parameter Count:** < 200,000 parameters.
* **Accuracy:** Achieve ≥ 85% accuracy on CIFAR-10.
* **Augmentation:** Use `albumentations` library (Horizontal Flip, ShiftScaleRotate, CoarseDropout).

---

## Model Architecture

The network follows a modular **C1-C2-C3-C4** architecture. Instead of standard pooling layers, we utilize strided convolutions to reduce spatial dimensions while increasing the receptive field.

### Convolution Types Used
To optimize the parameter count and receptive field, the network leverages:

**1. Dilated Convolution:**
Used to increase the receptive field exponentially without loss of resolution or coverage.


**2. Depthwise Separable Convolution:**
Used to reduce the total parameter count significantly compared to standard convolution by splitting the computation into a depthwise spatial convolution and a pointwise (1x1) convolution.


### Network Summary
* **Input:** 32x32x3 (CIFAR-10 Images)
* **Block 1:** Standard Conv + BatchNorm + ReLU
* **Block 2:** Depthwise Separable Conv + Strided Conv (Downsample)
* **Block 3:** Dilated Conv + Strided Conv (Downsample)
* **Block 4:** Standard Conv + GAP + Output Layer
* **Total Parameters:** < 200k (Optimized for efficiency)
* **Final Receptive Field:** > 44

---

## Data Augmentation (Albumentations)

We utilize the `albumentations` library to improve model generalization. The following pipeline is applied to the training data:

1.  **Horizontal Flip:** Randomized flipping.
2.  **ShiftScaleRotate:** Random affine transformations.
3.  **CoarseDropout:** Simulates occlusion (Cutout) to force the model to learn distributed features.

**Configuration:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    # Horizontal Flip
    A.HorizontalFlip(p=0.5),
    
    # ShiftScaleRotate
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    
    # CoarseDropout (Cutout)
    A.CoarseDropout(
        max_holes=1, max_height=16, max_width=16,
        min_holes=1, min_height=16, min_width=16,
        fill_value=[0.4914, 0.4822, 0.4465], # CIFAR-10 Mean
        mask_fill_value=None
    ),
    
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2(),
])

```

## Results:
Best Test Accuracy: >85%

Total Epochs: 50

Total Parameters: 195,126
