import pickle
from src.predict import PneumoniaPredictor  # Import before loading the pickle file

PICKLE_PATH = r"C:\Users\healthcare\cv\models\predictor1.pkl"

def main():
    # Load the saved predictor class
    with open(PICKLE_PATH, "rb") as file:
        predictor = pickle.load(file)  # Now Python knows the class location

    # Get user input for folder path
    folder_path = r"C:\Users\healthcare\cv\Sample images for testing"

    # Run pneumonia prediction
    predictor.predict_pneumonia_from_folder(folder_path)

if __name__ == "__main__":
    main()
