

import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.model import build_cnn
from src.utils import set_seed, prepare_mnist_splits, plot_history

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def main():
    set_seed(42)

    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_mnist_splits()
    num_classes = 10

    
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    
    datagen = ImageDataGenerator(rotation_range=10)
    datagen.fit(X_train)


    model = build_cnn(input_shape=(28, 28, 1), num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('models/best_model.h5', monitor='val_loss', save_best_only=True)
    ]

    
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=64),
        validation_data=(X_val, y_val_cat),
        epochs=20,
        callbacks=callbacks
    )

    
    model.save('models/final_model.h5')
    plot_history(history, out_path='outputs/training_history.png')
    print("Training complete! Best model saved.")


if __name__ == "__main__":
    main()
