# CNN and Conditional GAN on CIFAR-10

## Overview

This project explores both discriminative and generative deep learning models using the CIFAR-10 dataset. I designed and implemented a custom Convolutional Neural Network (CNN) for image classification and a Conditional Generative Adversarial Network (cGAN) for class-conditioned image generation, all built and trained in PyTorch.

The objective was to understand architectural design, adversarial training dynamics, conditioning mechanisms, and performance evaluation in both supervised and generative settings.

---

## Project Structure

CNN architecture:
- `convolution_1.py`

CNN training and testing:
- `main1.py`

Conditional GAN implementation:
- `cgan.py`

cGAN image plotting and visualization:
- `plot_cgan.py`

---

## Models Implemented

### 1. Convolutional Neural Network (CNN)

- Custom architecture designed from scratch
- Trained for multi-class image classification on CIFAR-10
- Optimized using cross-entropy loss
- Evaluated using accuracy and loss curves

Focus areas:
- Convolutional block design
- Feature hierarchy learning
- Optimization stability
- Performance evaluation

---

### 2. Conditional GAN (cGAN)

- Generator conditioned on class labels
- Discriminator trained on both images and labels
- Alternating adversarial optimization

Focus areas:
- Stable GAN training
- Label conditioning mechanism
- Generator-discriminator loss balancing
- Qualitative evaluation of generated samples

---

## Dataset

CIFAR-10:
- 60,000 32×32 RGB images
- 10 object classes
- Standard train/test split

---

## Technical Stack

- Python
- PyTorch
- NumPy
- Matplotlib

---

## Final Results for CNN

- **Final Training Accuracy:** 75.21%  
- **Final Test Accuracy:** 75.92%  
- **Final Training Loss:** 0.7237  
- **Final Test Loss:** 0.7346  

---

## Future Improvements

- Implement FID or Inception Score evaluation
- Experiment with deeper CNN architectures
- Implement Wasserstein GAN variants
- Apply data augmentation for improved robustness

---

## How to Run

To train and evaluate the CNN:
```
python main1.py
```

To train the Conditional GAN:
```
python cgan.py
```

To visualize generated images:
```
python plot_cgan.py
```
