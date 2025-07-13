# Contains utility functions that might be used across different modules
import tensorflow as tf
import numpy as np
import random
import os

def set_global_seeds(seed):
    """Set global seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Global seeds set to {seed}")

def check_data_dir(data_dir, target_classes):
    """Check if the data directory exists and contains the target classes."""
    print(f"Checking data directory: {data_dir} for classes: {target_classes}")
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not fount at {data_dir}")
        print(f"Please download the 'COVID-19 Radiography Dataset' from https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database and extract it to the specified path.")
        print(f"then update the DATA_DIR variable in config.py accordingly.")
        return False
    
    for cls in target_classes:
        images_path = os.path.join(data_dir, cls, 'images')
        if not os.path.exists(images_path):
            print(f"Error: Missing expected folder: {images_path}")
            print(f"Please ensure your dataset is extracted correctly with 'images' subfolders")
            return False
    
    print("Data directory and classes check passed.")
    return True