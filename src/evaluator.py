# Handles model evaluation on the test set and generates performance metrics and plots
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import config


def evaluate_model(model, test_generator, class_names):
    """
    Evaluates the model on the test set and prints detailed metrics.
    """
    print("Evaluating the model on the test set...")

    # Get predictions for detailed metrics
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_generator.samples // config.BATCH_SIZE + (test_generator.samples % config.BATCH_SIZE != 0))
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true = test_generator.classes[:len(y_pred_classes)] # Adjust length for potential incomplete last batch

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    #Classification Report
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("Classification Report:\n", report)

    # ROC Curve and AUC
    plt.figure(figsize=(10,8))
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=config.NUM_CLASSES)

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC cureve of {class_name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()