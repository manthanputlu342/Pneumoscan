# **Pneumonia Detection Project**  

## **Description**  
This project uses **Deep Learning (MobileNetV2, DenseNet121)** to detect **pneumonia from chest X-ray images**.  
It includes training, evaluating, and predicting pneumonia cases using a pre-trained model.  

---

## ** Project Structure**  

> **Note:** Modify directory paths according to your system.  

```
/cv_project/
│── main.py               # Runs prediction on images  
│── train.py              # Trains and saves the pneumonia detection model  
│── evaluate.py           # Evaluates the trained model  
│── model.py              # Defines the deep learning model architecture  
│── predict.py            # Loads the trained model and makes predictions  
│── data_preprocessing.py # Prepares the dataset for training/testing  
│── models/               # Stores trained model files (.keras, .pkl)  
│── data/                 # Contains training, validation, and test images  
│── requirements.txt      # Required dependencies  
│── README.md             # Project documentation  
```

---

## **⚙️ Installation & Setup**  

Ensure **Python 3.12.5** is installed.  
Then, install the dependencies using:  

pip install -r requirements.txt

---

## *Running the Project**  

### ** 1.Train the Model**  
To train the pneumonia detection model, run:  

```
python train.py
```
This will save the trained model inside the **`models/`** directory.  

---

### ** 2️.Evaluate the Model**  
To check model accuracy:  

```
python evaluate.py
```

---

### ** 3️.Make Predictions**  
Run the following command to detect pneumonia in new images:  

```
python main.py
```
This loads `predictor.pkl` and runs predictions.  

---

## **Important Notes**  
Ensure **`models/pneumonia_model1.keras`** exists before running predictions.  
Modify paths in **`main.py`** and **`predict.py`** if needed.  

---
