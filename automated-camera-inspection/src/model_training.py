"""
Model Training Module for Automated Camera Inspection System
Author: Dede Septa Maulana Fajar
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """
    Train the CNN model.
    """
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test))
    return history

def plot_training_history(history, output_dir="results"):
    """
    Plot training and validation accuracy/loss.
    """
    os.makedirs(output_dir, exist_ok=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.savefig(f"{output_dir}/accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(f"{output_dir}/loss.png")
    plt.close()
