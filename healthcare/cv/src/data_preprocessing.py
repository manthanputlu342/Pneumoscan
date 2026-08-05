import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

# Define dataset paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")
val_dir = os.path.join(DATA_DIR, "val")

# Data augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load training data
training_data = train_gen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=10, class_mode="binary", subset="training"
)

# Load validation data (FIX: Added class_mode="binary")
validation_data = train_gen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=10, class_mode="binary", subset="validation"
)

# Load test data
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=(224,224), batch_size=10, class_mode="binary", shuffle=False
)

