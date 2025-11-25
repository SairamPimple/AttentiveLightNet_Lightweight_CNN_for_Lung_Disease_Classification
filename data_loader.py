import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def enhance_and_prepare(image):
    """
    Applies CLAHE enhancement and scales image to [0, 1].
    Assumes input is a (224, 224, 1) NumPy array from the generator.
    """
    # Squeeze to 2D for cv2
    image_squeezed = image.squeeze()
    
    # Convert to uint8
    if image_squeezed.max() <= 1.0:
        image_uint8 = (image_squeezed * 255).astype('uint8')
    else:
        image_uint8 = image_squeezed.astype('uint8')
        
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image_uint8)
    
    # Scale back to [0, 1] float32
    image_float32 = enhanced_image.astype("float32") / 255.0
    
    # Expand dims back to (224, 224, 1) for Keras
    return np.expand_dims(image_float32, axis=-1)

def get_data_generators(base_dir, img_size=(224, 224), batch_size=32):
    """
    Creates and returns train, validation, and test data generators.
    """
    target_size = img_size

    # Training Data Generator (with Augmentation)
    train_datagen = ImageDataGenerator(
        preprocessing_function=enhance_and_prepare,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True
    )

    # Validation/Test Data Generator (No Augmentation)
    test_datagen = ImageDataGenerator(
        preprocessing_function=enhance_and_prepare
    )

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    print(f"Loading train data from {train_dir}...")
    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    print(f"Loading validation data from {val_dir}...")
    val_gen = test_datagen.flow_from_directory(
        directory=val_dir,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Loading test data from {test_dir}...")
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen