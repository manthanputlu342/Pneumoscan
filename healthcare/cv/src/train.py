from model import build_model
from data_preprocessing import training_data, validation_data
import os
print(os.path.exists(r"C:\Users\vansh\Desktop\cv\models\pneumonia_model1.keras"))  # Check if the file exists

# Build and train model
model = build_model()
history = model.fit(training_data, epochs=7, validation_data=validation_data)

# Save trained model
model.save(r"C:\Users\vansh\Desktop\cv\models\pneumonia_model2.keras")
print("✅ Model saved successfully!")
