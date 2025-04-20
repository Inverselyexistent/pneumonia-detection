import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define paths
base_dir = 'chest_xray'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

# Parameters
IMG_SIZE = (224, 224)  # Standard size for CNNs
BATCH_SIZE = 32

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=20,  # Rotate up to 20 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip horizontally
    fill_mode='nearest',  # Fill gaps with nearest pixel
    brightness_range=[0.8, 1.2],  # Adjust brightness
    validation_split=0.2  # Use 20% of train data for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='training'  # 80% of train data
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='validation'  # 20% of train data for validation
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale'
)

# Visualize preprocessed samples
sample_images, sample_labels = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')  # Squeeze to remove channel dim
    plt.title(f'Label: {sample_labels[i]}')
    plt.axis('off')
plt.savefig('preprocessed_sample.png')
print("Preprocessed sample saved as preprocessed_sample.png")