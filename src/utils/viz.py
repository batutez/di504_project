"""
Plotting utilities, confusion matrices, per class ROC curves, training and validation loss curves.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Render a confusion matrix as a seaborn heatmap.

    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true, y_proba, class_names, title="ROC Curves", save_path=None):
    """
    Plot one vs rest (ovr) ROC curves, one line per class, on a single figure.
    If provided, function saves the curve in a desired path.

    """
    y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses, val_losses, title="Training Curve", save_path=None):
    """
    Plot train versus validation loss over epochs. This is for GRU and ALBERT cases.
    If provided, function saves the curve in a desired path.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()