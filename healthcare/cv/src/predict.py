import pickle
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


class PneumoniaPredictor:
    def __init__(self, model_path=r"C:\Users\vansh\Desktop\cv\models\pneumonia_model1.keras"):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"🚨 Model file not found at: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def predict_pneumonia_from_folder(self, folder_path):
        """Predicts pneumonia for all images in a folder."""

        # Ensure folder exists
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return

        # List all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print("❌ No valid image files found in the folder.")
            return

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)

            try:
                # Load and preprocess the image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)  # Required for DenseNet121

                # Make prediction
                prediction = self.model.predict(img_array)[0][0]

                # Display Probability
                probability = prediction * 100  # Convert to percentage

                # Print results
                print(f"🖼️ **Image:** {img_name}")
                print(f"🩺 Pneumonia Probability: {probability:.2f}%")

                # Final Result
                if prediction > 0.5:
                    print("🔴 **Pneumonia Detected**\n")
                else:
                    print("🟢 **Normal Lungs**\n")

                # Show image
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Pneumonia: {probability:.2f}%")
                plt.show()

            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")


# ✅ Save the class instance as a pickle file
os.makedirs("models", exist_ok=True)  # Ensure models directory exists
predictor = PneumoniaPredictor()  # Create an instance

with open(r"C:\Users\vansh\Desktop\cv\models\predictor1.pkl", "wb") as file:
    pickle.dump(predictor, file)

print("✅ Predictor function saved as predictor.pkl!")
