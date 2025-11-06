

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import prepare_mnist_splits, plot_confusion_matrix

class_names = [str(i) for i in range(10)]  


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_mnist_splits()

    model = load_model('models/best_model.h5')
    print("Loaded best_model.h5")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = np.mean(y_pred == y_test)
    print(f"📊 Test Accuracy: {acc*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, out_path='outputs/confusion_matrix.png')


if __name__ == "__main__":
    main()
