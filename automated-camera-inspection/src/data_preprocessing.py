"""
Data Preprocessing Module for Automated Camera Inspection System
Author: Dede Septa Maulana Fajar
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_images(data_dir, img_size=(128, 128)):
    """
    Load and preprocess images from a directory.
    """
    X, y = [], []
    classes = os.listdir(data_dir)

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(idx)

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)
    return X, y, classes

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
