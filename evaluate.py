import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_loader import get_data_generators
# Import custom components for loading
from model_architecture import swish_plus, DynamicDropout, CustomFocalLoss

# --- Configuration ---
MODEL_PATH = 'models/attentive_lightnet.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'data'

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 1. Load Data
    _, _, test_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    class_labels = list(test_gen.class_indices.keys())

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    # Note: @register_keras_serializable handles custom objects automatically
    model = load_model(MODEL_PATH)

    # 3. Evaluate
    print("Evaluating on test data...")
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # 4. Predict
    print("Generating predictions...")
    test_gen.reset() # Ensure order matches labels
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    # 5. Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    plot_confusion_matrix(y_true, y_pred, class_labels)

if __name__ == "__main__":
    evaluate()