"""
Inference Module for Automated Camera Inspection System
Author: Dede Septa Maulana Fajar
"""

import cv2
import numpy as np
import tensorflow as tf

def load_trained_model(model_path):
    """
    Load a trained Keras model.
    """
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, img_size=(128,128)):
    """
    Preprocess single image for inference.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_image(model, image_path, class_names, img_size=(128,128)):
    """
    Run inference on a single image.
    """
    img = preprocess_image(image_path, img_size)
    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)
    return class_names[class_id], confidence
