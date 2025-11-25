import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from model_architecture import build_attentive_lightnet, DynamicDropout, CustomFocalLoss
from data_loader import get_data_generators

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 30
MODEL_DIR = 'models'
MODEL_NAME = 'attentive_lightnet.keras'
DATA_DIR = 'data'

# --- Custom Callback ---
class DynamicDropoutCallback(tf.keras.callbacks.Callback):
    """Callback to update the DynamicDropout layer's rate each epoch."""
    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, DynamicDropout):
                layer.update_rate(epoch)

# --- Plotting Function ---
def plot_history(history, save_path='training_history.png'):
    """Plots and saves training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.suptitle('Training and Validation History')
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    # 1. Data
    train_gen, val_gen, _ = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # 2. Class Weights
    print("Calculating class weights...")
    y_train_int = train_gen.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # 3. Model
    print("Building model...")
    model = build_attentive_lightnet(input_shape=IMG_SIZE+(1,), num_classes=2, total_epochs=NUM_EPOCHS)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4), # 1e-4 as per typical fine-tuning
        loss=CustomFocalLoss(gamma=2.0),
        metrics=['accuracy']
    )
    model.summary()

    # 4. Callbacks (Updated per reference)
    model_save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        DynamicDropoutCallback()
    ]

    # 5. Training
    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=NUM_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights_dict
    )
    
    print(f"Training complete. Best model saved to {model_save_path}")
    plot_history(history)

if __name__ == "__main__":
    train()