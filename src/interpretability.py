#Contains the Grad-CAM logic and visualization functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
import config

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given image array and model.
    
    Args:
        img_array: Preprocessed image array.
        model: Keras model.
        last_conv_layer_name: Name of the last convolutional layer in the model.
        pred_index: Index of the predicted class (optional).
    
    Returns:
        Heatmap as a numpy array.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()


def display_gradcam(img_path, model, last_conv_layer_name, class_names):
    """
    Loads an image, generates a Grad-CAM heatmap, and displays it superimposed.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img_array = img_array / 255.0

    predictions = model.predict(processed_img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class]*100

    heatmap = make_gradcam_heatmap(processed_img_array, model, last_conv_layer_name, pred_index=predicted_class)

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        # Fallback for images not readable by cv2 directly (e.g., some PNGs)
        if img_cv.ndim == 2: # grayscale to BGR
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        else: # RGB to BGR
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    else:
        img_cv = cv2.resize(img_cv, config.IMG_SIZE)

    heatmap = cv2.resize(heatmap, config.IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_cv # alpha = 0.4 from config
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image\nPredicted: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()


def show_sample_gradcams(model, test_df, class_names, num_examples=5):
    """
    Displays Grad-CAM heatmaps for a few random samples from the test set.
    
    Args:
        model: Keras model.
        test_df: DataFrame containing test image paths and labels.
        class_names: List of class names.
        num_examples: Number of examples to display.
    """
    print(f"\n Displaying {num_examples} Grad-CAM examples (True label & Prediction):")
    sample_test_df = test_df.sample(n=num_examples, random_state=config.SEED)

    for index, row in sample_test_df.iterrows():
        img_path = row['path']
        true_label_name = row['label']

        print(f"\n -- Sample from test_df (Index: {index}) --")
        print(f"True Class: {true_label_name}")

        display_gradcam(img_path, model, config.LAST_CONV_LAYER_NAME_EFFICIENTNETB0, class_names)