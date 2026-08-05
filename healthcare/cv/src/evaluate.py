import os
import tensorflow as tf
from data_preprocessing import test_data

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "cv/models", "pneumonia_model2.keras")

model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate on test data
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.2f}, Loss: {loss:.2f}")
