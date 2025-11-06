

import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split



def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def prepare_mnist_splits(seed=42):
    """
    Loads MNIST (28x28 grayscale images of digits 0–9),
    normalizes them, and splits into train/val/test.
    """
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)


    X = X.astype("float32") / 255.0


    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    print("Dataset split completed")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_history(history, out_path="outputs/training_history.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved training history → {out_path}")


def plot_confusion_matrix(cm, classes, out_path="outputs/confusion_matrix.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8,
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix → {out_path}")
