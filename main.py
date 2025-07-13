import tensorflow as tf
import config
from src import utils, data_loader, model_builder, trainer,  evaluator,interpretability
import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

def main():    

    # *** Setup *** Set global seeds for reproducibility
    print("Starting the COVID-19 Chest X-Ray Image Classification Tutorial...")    
    utils.set_global_seeds(config.SEED)

    # *** Validate *** Check if the data directory is valid
    if not utils.check_data_dir(config.DATA_DIR, config.TARGET_CLASSES):
        return # Exit if data directory is invalid or not setup properly
    

    # *** Load Data *** Load image paths and create dataset splits
    print("step 1: Loading image paths and creating dataset splits..")
    try:
        df = data_loader.load_image_paths(config.DATA_DIR, config.TARGET_CLASSES)
        print(f"Total images found: {len(df)}")
        if len(df) == 0:
            print("Error: No images found in the dataset directory. Please check the path and ensure images are present.")
            return
        
        train_gen, val_gen, test_gen, class_names, test_df_for_gradcam = data_loader.create_data_generators(
        df, config.IMG_SIZE, config.BATCH_SIZE, config.TARGET_CLASSES, config.SEED)

        print("Data generators created successfully.")

        # Get the actual labels used in the training generator
        y_train_labels_numerical = train_gen.classes
        
        # Get unique class names from the generator's internal mapping
        # Sort these to ensure consistency with class_weight.compute_class_weight
        generator_class_names = sorted(train_gen.class_indices.keys())
        
        # Compute weights
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_labels_numerical), # Use unique numerical labels from the training data
            y=y_train_labels_numerical
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights_array)}
        
        print(f"Computed class weights (numerical index: weight): {class_weight_dict}")

    except Exception as e:
        print(f"Error during data loading or generator creation: {e}")
        return # Exit on error

    # 3. Build and train the model
    print("\nStep 2: Building and training the model...")
    model = None # Initialize model to None
    base_model = None # Initialize base_model to None
    try:
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"Found existing model at {config.MODEL_SAVE_PATH}. Loading it. To retrain, delete the file.")
            model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
            # Try to get base_model (EfficientNetB0) by name or by assuming it's the first layer
            if 'efficientnetb0' in model.layers[0].name: # Adjust name if necessary
                 base_model = model.layers[0]
            else: # Fallback: iterate layers to find it
                for layer in model.layers:
                    if 'efficientnetb0' in layer.name:
                        base_model = layer
                        break
                if base_model is None:
                    print("Warning: Could not identify EfficientNetB0 base_model after loading. Grad-CAM might fail.")

        else:
            print("No existing model found. Building a new model.")
            model, base_model = model_builder.build_cnn_model(config.IMG_SIZE, config.NUM_CLASSES, config.LEARNING_RATE)
            print("Model built. Starting training...")
            history = trainer.train_model(model, train_gen, val_gen, class_weight=class_weight_dict) # <-- NEW: Pass class_weight_dict
            trainer.plot_training_history(history)
            model = tf.keras.models.load_model(config.MODEL_SAVE_PATH) # Load the best model after training
        print("Model ready.")
    except Exception as e:
        print(f"Error during model building or training: {e}")
        return

    # 4. Evaluate the model
    print("\nStep 3: Evaluating the model...")
    try:
        evaluator.evaluate_model(model, test_gen, class_names)
        print("Model evaluation complete.")
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    # 5. Interpret the model with Grad-CAM
    print("\nStep 4: Interpreting the model with Grad-CAM...")
    try:
        interpretability.show_sample_gradcams(model, test_df_for_gradcam, class_names, num_examples=5)
        print("Grad-CAM interpretation complete.")
    except Exception as e:
        print(f"Error during Grad-CAM interpretation: {e}")
        # Do not return here, as the rest of the script might still be useful
        # You might want to remove this 'return' in a tutorial to show the error
        # but allow the script to finish without crashing if Grad-CAM is the only issue.


    print("\nTutorial execution finished.")

    

if __name__ == "__main__":
    main()