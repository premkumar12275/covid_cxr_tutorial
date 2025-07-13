# Handles loading image paths, creating the DataFrame, and setting up ImageDataGenerators for train, validation, and test sets.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import config

def load_image_paths(data_dir, target_classes):
    """
    Collects all image paths and their corrsponding labels into a pandas DataFrame.
    """
    image_paths = []
    labels = []

    print(f"Loading image paths from {data_dir}...")

    for class_name in target_classes:
        class_images_dir = os.path.join(data_dir, class_name, 'images')
        for img_name in os.listdir(class_images_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_images_dir, img_name))
                labels.append(class_name)

        df = pd.DataFrame({
            'path': image_paths,
            'label': labels
        })
    return df

def create_data_generators(df, img_size, batch_size, target_classes, seed):
    """
    Splits the DataFrame into tran/Validation/test sets and creates ImageDataGenerators.
    """
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    #store class names in the correct order based on encoding
    class_names = [label_encoder.inverse_transform([i])[0] for i in range(len(target_classes))]

    #stratified split into train and validation sets
    tran_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label_encoded'], random_state=seed)
    train_df, val_df = train_test_split(tran_val_df, test_size=0.176, stratify=tran_val_df['label_encoded'], random_state=seed)


    print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}, Test images: {len(test_df)}")

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255) # only rescale

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='path', y_col='label', target_size=img_size,
        color_mode='rgb', batch_size=batch_size, class_mode='categorical', shuffle=True, seed=seed
    )

    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='path', y_col='label', target_size=img_size,
        color_mode='rgb', batch_size=batch_size, class_mode='categorical', shuffle=False, seed=seed
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='path', y_col='label', target_size=img_size,
        color_mode='rgb', batch_size=batch_size, class_mode='categorical', shuffle=False, seed=seed
    )

    return train_generator, validation_generator, test_generator, class_names, test_df # Return test_df for Grad-CAM paths later


def compute_class_weights(df, target_classes): # <-- NEW FUNCTION
    """
    Computes class weights for imbalanced datasets.
    """
    # Ensure labels are numerical for class_weight.compute_class_weight
    # Use the labels from the full dataframe before splitting to get overall distribution
    # We'll use the 'label' column directly, which is strings
    
    # Map target_classes to numerical indices for consistent ordering
    # It's crucial that the order of class_names passed to the generator matches the order
    # class_weight.compute_class_weight uses. The generator uses alphabetical order by default
    # unless you explicitly map class_indices.
    # Let's rely on LabelEncoder for consistent mapping to integers 0, 1, 2...
    
    label_encoder = LabelEncoder()
    # Fit on all target classes to ensure consistency, even if not all appear in df (unlikely here)
    label_encoder.fit(target_classes) 
    
    # Now transform the labels from the DataFrame
    numerical_labels = label_encoder.transform(df['label'])
    
    # Compute weights
    # unique_labels should be the actual numerical labels that exist in the data
    # y should be the numerical labels of the data
    
    # Use the labels from the training set only for class weight calculation,
    # as the weights are applied to the training loss.
    # To get the training labels, we'd need to modify create_data_generators to return train_df_labels,
    # or re-extract them. For simplicity, let's calculate based on the full initial df.
    # However, for true best practice, you'd calculate based *only* on your training set's labels.
    # Let's adjust create_data_generators to return the train_df for this purpose.

    # Re-evaluating: it's better to compute weights based on the actual `train_df`
    # returned by `create_data_generators`
    print("Class weights computation moved to main.py after train_df is available for accuracy.")
    return {} # This function will now be empty or removed, as weights will be computed in main.py