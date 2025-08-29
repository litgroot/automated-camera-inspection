"""
Evaluation Module for Automated Camera Inspection System
Author: Dede Septa Maulana Fajar
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def evaluate_model(model, X_test, y_test, class_names, output_dir="results"):
    """
    Evaluate model with classification report and confusion matrix.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("Classification Report:\n", report)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
