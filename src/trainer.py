import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import config

def train_model(model, train_generator, validation_generator, class_weight=None):
    """
    Trains the given keras model using provided data generators and callbacks

    """

    print("Traing the model...")

    checkpoint = ModelCheckpoint(
        config.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.EPOCHs,
        steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
        validation_steps=validation_generator.samples // config.BATCH_SIZE,
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weight
    )

    print("Model training completed.")
    return history


def plot_training_history(history):
    """
    Plots training and validation accuracy and loss.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy') 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()