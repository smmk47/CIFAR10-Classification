# Vision Transformer, Hybrid CNN-MLP, and ResNet for CIFAR-10 Image Classification

This repository presents the implementation and evaluation of three deep learning architectures—**Vision Transformer (ViT)**, **Hybrid CNN-MLP**, and **ResNet**—for **image classification** on the **CIFAR-10** dataset. The project compares the performance of these models based on key metrics like **accuracy**, **precision**, **recall**, **F1-score**, **training time**, **memory usage**, and **inference speed**.

The **Vision Transformer (ViT)** outperforms the other two models, demonstrating the effectiveness of Transformer-based architectures in image classification tasks. The repository contains the full code for training, evaluation, and comparison of these architectures.



## Introduction
Image classification is a crucial task in computer vision, applied to domains such as autonomous driving, medical diagnosis, and surveillance. Traditionally, **Convolutional Neural Networks (CNNs)** have been the go-to model for image classification due to their ability to learn spatial hierarchies. However, recent advances have introduced **Transformer-based architectures**, such as the **Vision Transformer (ViT)**, which apply self-attention mechanisms to model long-range dependencies in images.

In this project, three different architectures are evaluated for CIFAR-10 image classification:
1. **Vision Transformer (ViT)** - A state-of-the-art Transformer-based model for image classification.
2. **Hybrid CNN-MLP** - A hybrid architecture combining CNNs for feature extraction and MLPs for classification.
3. **ResNet** - A widely used CNN architecture known for its residual connections and ability to train deep networks efficiently.

This comparative analysis helps to understand the strengths and limitations of each architecture and provides insights into their suitability for image classification tasks.

## Dataset
The **CIFAR-10 dataset** is a well-known dataset used for training machine learning models. It contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into:
- **Training Set**: 50,000 images.
- **Test Set**: 10,000 images.

The dataset consists of 10 categories:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Preprocessing
- **Normalization**: Images are normalized to a range of [0, 1].
- **Data Augmentation**: Random horizontal flips, rotations, and color jittering are applied to the training data to improve model robustness.

## Model Architectures
### Vision Transformer (ViT)
The **Vision Transformer (ViT)** applies the Transformer architecture to image classification tasks. The model operates by dividing images into patches, linearly embedding them, and applying Transformer layers with multi-head self-attention. The ViT model has the following key components:
- **Patch Embeddings**: The image is split into non-overlapping patches, which are then flattened and projected into a lower-dimensional space.
- **Multi-Head Self-Attention**: Self-attention allows the model to capture dependencies between different parts of the image.
- **MLP Head**: The final classification is performed using a multi-layer perceptron (MLP).

### Hybrid CNN-MLP
The **Hybrid CNN-MLP** model combines a Convolutional Neural Network (CNN) for feature extraction and a Multi-Layer Perceptron (MLP) for classification:
- **CNN Layers**: Extract high-level features from images using convolution, pooling, and batch normalization layers.
- **MLP Layers**: After feature extraction, the CNN output is passed through fully connected layers to classify the image.

### ResNet
**ResNet (Residual Network)** is a deep CNN architecture known for its residual connections, which help mitigate the vanishing gradient problem during training. The key features of ResNet are:
- **Residual Blocks**: These allow for the training of very deep networks by enabling gradient flow through skip connections.
- **Pretrained ResNet**: For this project, we use a pretrained ResNet model with transfer learning to fine-tune it for the CIFAR-10 dataset.

## Training Details
- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 0.001 (for Adam optimizer)
- **Optimizer**: Adam optimizer
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### Hyperparameter Tuning
Hyperparameters such as learning rate, batch size, and number of layers were tuned for each architecture to achieve optimal performance.

## Evaluation
The models were evaluated using the following metrics:
- **Accuracy**: Measures the proportion of correctly classified images.
- **Precision**: Measures the proportion of true positive predictions out of all positive predictions.
- **Recall**: Measures the proportion of true positive predictions out of all actual positive instances.
- **F1-Score**: Harmonic mean of precision and recall, providing a balanced measure of performance.

## Results
The performance of the three models was compared on the following metrics:
- **Accuracy**: ViT demonstrated superior accuracy compared to Hybrid CNN-MLP and ResNet.
- **Training Time**: ViT and ResNet had faster training times than the Hybrid CNN-MLP model.
- **Memory Usage**: ViT required more memory due to its attention mechanism, while ResNet was more memory efficient.
- **Inference Speed**: ResNet had the fastest inference speed, followed by Hybrid CNN-MLP and ViT.


