"""
model.py
Lightweight CNN for ASL Alphabet (29 classes)
Francesco Benincasa – MIT Licence
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(200, 200, 3),
                num_classes=29,
                dropout_rate=0.5,
                kernel_size=3):
    """
    Returns a compiled Keras model.
    """
    model = models.Sequential([
        layers.Conv2D(32, kernel_size, padding='same', activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, kernel_size, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, kernel_size, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model