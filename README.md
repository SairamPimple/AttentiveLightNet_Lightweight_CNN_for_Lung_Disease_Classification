Efficient Lung Disease Detection Using AttentiveLightNet
A Novel Lightweight CNN with Attention and Adaptive Dropout
This repository contains the implementation of AttentiveLightNet, a novel deep learning model designed for highly efficient and accurate binary classification of lung diseases (Normal vs. Pneumonia) from chest X-ray images.

The project emphasizes balancing high diagnostic performance with minimal computational overhead, making the model suitable for deployment in resource-constrained environments, such as mobile or edge devices in clinics.

✨ Key Features of AttentiveLightNet
AttentiveLightNet achieves its efficiency and accuracy through several innovative architectural and training elements:

Lightweight Architecture: Uses Depthwise Separable Convolutions and a custom Multi-Branch Inception Module to significantly reduce the total number of parameters.

Squeeze-and-Excitation (SE) Attention Blocks: Integrated after each module to perform channel-wise feature recalibration, helping the model focus adaptively on the most diagnostically relevant features (e.g., subtle opacities).

Swish+ Activation Function: A custom, scaled non-linearity defined as x⋅σ(1.5x) to promote smoother gradient flow and stable training convergence.

Dynamic Dropout Regularization: An adaptive regularization strategy where the dropout rate is high in early epochs and gradually decays, preventing early overfitting and encouraging better generalization.

Optimized Pre-processing: Includes Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve localized contrast in grayscale chest X-rays, enhancing the visibility of subtle pulmonary indicators.

🏗️ Model Architecture Overview
AttentiveLightNet is a custom Convolutional Neural Network (CNN) built on the Keras Functional API.

Stage

Key Components

Function

Output Channels

Parameters

Initial Layer

Conv2D (stride 2) + MaxPool

Downsample and extract low-level features.

16

Low

Feature Blocks (x3)

Inception + SE Block + MaxPool

Extract multi-scale features and apply channel attention.

32, 64, 128 (Progressive)

Moderate

Classification Head

Global Avg Pooling + Dense Layers (Swish+, Dropout)

Convert spatial features to a fixed-length vector and generate class probabilities.

128, 64, 2 (Output)

Low

Total Trainable Parameters: ≈345,266

📊 Performance and Results
The model was trained and validated on a publicly available Chest X-Ray Images (Pneumonia) dataset.

Performance on Test Set (Normal vs. Pneumonia)
Metric

Result

Note

Test Accuracy

97.3%

Overall model robustness.

Test Precision (Pneumonia)

0.98

High confidence in positive predictions.

Test Recall (Pneumonia)

0.98

Strong ability to identify true positive cases.

Test AUC

0.9859

Excellent discriminative power.

Efficiency Comparison
AttentiveLightNet achieved accuracy competitive with much larger benchmark models (like ResNet50V2 and DenseNet121) while using significantly fewer parameters, making it highly resource-efficient.

Model

Test Accuracy

Model Parameters (Millions)

AttentiveLightNet

93.43%

0.345 M

EfficientNetB0

93.26%

4.2 M

DenseNet121

94.71%

7.3 M

ResNet50V2

95.65%

24.0 M

💻 Setup and Usage
Prerequisites
Python 3.8+

TensorFlow / Keras 2.x

NumPy, Pandas, Scikit-learn

OpenCV (cv2) for image pre-processing.

Installation
# It is recommended to use a virtual environment
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python

Data Preparation
Download Dataset: Obtain the Chest X-Ray Images (Pneumonia) dataset (commonly available on Kaggle).

Organize: Ensure your data is organized into train/, val/, and test/ directories, with subdirectories for Normal and Pneumonia cases.

Run Pre-processing: The included data handling scripts (similar to those in APPENDIX-1) perform:

Grayscale conversion.

Resizing to 224x224.

CLAHE enhancement.

Stratified splitting and augmentation (training set only).

Training the Model
The core model definition is found in the implementation details (similar to APPENDIX-2). Training should use the following configuration:

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Callbacks: Early Stopping (patience=5), ReduceLROnPlateau (patience=3)

Epochs: 30 (with mixed-precision training enabled)

🎓 Thesis Details
This project was submitted in partial fulfillment of the requirements for the degree of Master of Technology in Computer Science and Engineering (Big Data Analytics).

Author: Sairam Pimple (23MCB1006)

Institution: Vellore Institute of Technology, Chennai

Date: April, 2025

Supervised by: Dr. Mansoor Hussain

🚀 Future Work
Potential areas for improvement and expansion of the AttentiveLightNet framework:

Multi-Class Extension: Expand the model to classify a wider range of lung conditions (e.g., COVID-19, Tuberculosis, Opacity) beyond the current binary task.

Model Quantization: Apply post-training quantization to reduce the model size further for ultra-low-resource edge deployment.

Explainable AI (XAI): Integrate methods like Grad-CAM to generate visual heatmaps, improving clinical interpretability and trust.
