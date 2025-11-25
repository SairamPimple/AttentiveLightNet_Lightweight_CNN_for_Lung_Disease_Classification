# AttentiveLightNet: Pneumonia Detection from X-Rays

This is an end-to-end deep learning project that builds, trains, and deploys a custom Convolutional Neural Network (CNN) to classify chest X-ray images as "Normal" or "Pneumonia."

The core of this project is a novel architecture named **AttentiveLightNet**, which incorporates modern deep learning techniques to achieve high accuracy on an imbalanced dataset.

**Live Demo:** [Link to your hosted Gradio app on Hugging Face Spaces]

![Demo GIF](link-to-your-demo-gif.gif) ## Contents
- [Problem Statement](#problem-statement)
- [Model Architecture](#model-architecture-attentivelightnet)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)

## Problem Statement
Pneumonia is a life-threatening respiratory infection that inflames the air sacs in one or both lungs. Early and accurate diagnosis from radiological images like X-rays is crucial for timely treatment. This project aims to build an accurate and efficient classifier to assist radiologists by identifying signs of pneumonia in chest X-rays.

## Model Architecture: AttentiveLightNet

The model is a custom-built CNN designed for high performance and efficiency. It is not a standard VGG or ResNet but a unique architecture combining several advanced concepts:

**Custom Activation (Swish+)**: Uses a modified swish_plus activation (x * sigmoid(1.5 * x)) for better non-linearity.

**Inception-style Modules**: Employs parallel 1x1, 3x3, and 5x5 separable convolutions to capture features at multiple scales.

**Squeeze-and-Excitation (SE Blocks)**: Each Inception module is followed by an se_block (an attention mechanism) that re-calibrates channel-wise feature responses, allowing the model to focus on the most informative features.

**Dynamic Dropout**: A custom DynamicDropout layer that linearly increases its dropout rate from 0.1 to 0.5 as training progresses, acting as a powerful regularizer.

**Custom Preprocessing (CLAHE)**: All images are preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast in the X-rays before being fed to the network.

**Focal Loss**: The model is trained using CustomFocalLoss to effectively handle the severe class imbalance between "Normal" and "Pneumonia" images.

## Project Structure
```
pneumonia-classifier/
├── models/
│   └── attentive_lightnet.keras     # Trained model
├── data/
│   └── (Kaggle dataset)
├── app.py                           # The Streamlit web app
├── train.py                         # Script to train the model
├── evaluate.py                      # Script to evaluate the model
├── model_architecture.py            # "AttentiveLightNet" definition
├── data_loader.py                   # Data loading & preprocessing (with CLAHE)
├── requirements.txt                 # Dependencies
└── README.md                        # This file
├── training_history.png             # Output from train.py
└── confusion_matrix.png             # Output from evaluate.py
```

## How to Run

### 1. Setup
Clone the repository and install the required packages:
```bash
git clone [https://github.com/your-username/pneumonia-classifier.git](https://github.com/your-username/pneumonia-classifier.git)
cd pneumonia-classifier
pip install -r requirements.txt
```
Download the [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place the `train`, `test`, and `val` folders inside the `data/` directory.

### 2. Training
To train the model from scratch, run the training script. This will use mixed precision, apply all callbacks, and save the final model to the `models/` folder.

```bash
python train.py
```

### 3. Evaluation
To evaluate the trained model on the test set, run:
```bash
python evaluate.py
```

### 4. Launching the Web App
To start the interactive Gradio demo, run:
```bash
streamlit run app.py
```
This will launch a local web server. You can also upload this script to [Hugging Face Spaces](https://huggingface.co/spaces) for a free, shareable public demo.

## Results
The model was trained on an imbalanced dataset, using class weights to compensate. It achieved **82.7% accuracy** on the held-out test set.

### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.86      0.65      0.74       234
   Pneumonia       0.81      0.94      0.87       390

    accuracy                           0.83       624
   macro avg       0.84      0.79      0.80       624
weighted avg       0.83      0.83      0.82       624
```

### Confusion Matrix
![Confusion Matrix](link-to-your-cm-image.png) ### Training & Validation Curves
![Loss/Accuracy](link-to-your-loss-acc-image.png) ````

---


