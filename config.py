import os


# General configuration
IMG_SIZE = (256, 256)  # Image size for input images
BATCH_SIZE = 32  # Batch size for training and inference
EPOCHs = 100
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
SEED = 42  # Random seed for reproducibility

# Dataset Configuration
DATA_DIR = 'C:\personal\MSAI\AI_HC\SelfLearingTutorial\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset'

# since we do not have train and test folders we will use manual split
TARGET_CLASSES = ['COVID', 'Lung_Opacity','Normal', 'Viral Pneumonia']  
NUM_CLASSES = len(TARGET_CLASSES)

# Model Configuration
MODEL_SAVE_PATH = 'best_model_covid_simple.keras'
LAST_CONV_LAYER_NAME_EFFICIENTNETB0 ='block7a_project_bn' # Common last conv layer for Grad-CAM
# we have to verify this by running model.summary() in `model_builder.py` and checking the last Conv2D layer 

# Training callbakcs configuration
PATIENCE = 20  # Early stopping patience