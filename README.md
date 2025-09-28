# Efficient Lung Disease Classification Using AttentiveLightNet  
### A Novel Lightweight CNN with Attention and Adaptive Dropout  

This repository contains the implementation of **AttentiveLightNet**, a novel deep learning model designed for highly efficient and accurate binary classification of lung diseases (**Normal vs. Pneumonia**) from chest X-ray images.  

The project emphasizes balancing high diagnostic performance with minimal computational overhead, making the model suitable for deployment in resource-constrained environments, such as mobile or edge devices in clinics.  

---

## ✨ Key Features of AttentiveLightNet  

AttentiveLightNet achieves its efficiency and accuracy through several innovative architectural and training elements:  

- **Lightweight Architecture**: Uses Depthwise Separable Convolutions and a custom Multi-Branch Inception Module to significantly reduce the total number of parameters.  
- **Squeeze-and-Excitation (SE) Attention Blocks**: Integrated after each module to perform channel-wise feature recalibration, helping the model focus adaptively on the most diagnostically relevant features (e.g., subtle opacities).  
- **Swish+ Activation Function**: A custom, scaled non-linearity defined as `x⋅σ(1.5x)` to promote smoother gradient flow and stable training convergence.  
- **Dynamic Dropout Regularization**: An adaptive regularization strategy where the dropout rate is high in early epochs and gradually decays, preventing early overfitting and encouraging better generalization.  
- **Optimized Pre-processing**: Includes Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve localized contrast in grayscale chest X-rays, enhancing the visibility of subtle pulmonary indicators.  

---

## 🏗️ Model Architecture Overview  

AttentiveLightNet is a custom **Convolutional Neural Network (CNN)** built on the Keras Functional API.  

| Stage             | Key Components                          | Function                                                     | Output Channels | Parameters |
|-------------------|------------------------------------------|-------------------------------------------------------------|-----------------|------------|
| Initial Layer     | Conv2D (stride 2) + MaxPool              | Downsample and extract low-level features.                   | 16              | Low        |
| Feature Blocks (x3)| Inception + SE Block + MaxPool           | Extract multi-scale features and apply channel attention.    | 32, 64, 128     | Moderate   |
| Classification Head| Global Avg Pooling + Dense (Swish+, Dropout) | Convert spatial features to vector & generate probabilities. | 128, 64, 2      | Low        |  

**Total Trainable Parameters: ≈ 345,266**  

---

## 📊 Performance and Results  

The model was trained and validated on a publicly available **Chest X-Ray Images (Pneumonia)** dataset.  

### Performance on Test Set (Normal vs. Pneumonia)  

| Metric                  | Result   | Note                                   |
|--------------------------|----------|----------------------------------------|
| Test Accuracy            | 97.3%    | Overall model robustness.              |
| Test Precision (Pneumonia)| 0.98    | High confidence in positive predictions. |
| Test Recall (Pneumonia)  | 0.98     | Strong ability to identify positives.  |
| Test AUC                 | 0.9859   | Excellent discriminative power.        |  

### Efficiency Comparison  

AttentiveLightNet achieved accuracy competitive with much larger benchmark models while using **significantly fewer parameters**.  

| Model            | Test Accuracy | Model Parameters (Millions) |
|------------------|---------------|------------------------------|
| **AttentiveLightNet** | **93.43%**    | **0.345 M**                   |
| EfficientNetB0   | 93.26%        | 4.2 M                        |
| DenseNet121      | 94.71%        | 7.3 M                        |
| ResNet50V2       | 95.65%        | 24.0 M                       |  

---

## 💻 Setup and Usage  

### Prerequisites  
- Python 3.8+  
- TensorFlow / Keras 2.x  
- NumPy, Pandas, Scikit-learn  
- OpenCV (cv2) for image pre-processing.  

### Installation  
```bash
# It is recommended to use a virtual environment
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python
