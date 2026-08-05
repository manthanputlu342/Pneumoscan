#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras 
import tensorflow as tf
import cv2
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
import os 
import glob
import cv2
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping


import warnings 
warnings.filterwarnings("ignore")


# In[91]:


# dataset directory
train_dir = r"C:\Users\vansh\Downloads\chest_xray\train"
test_dir = r"C:\Users\vansh\Downloads\chest_xray\test"


# In[110]:


# image datagenerator
# Data augmentation and preprocessing

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

#train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2,)

training_data = train_gen.flow_from_directory(directory=train_dir,
                                             target_size = (224,224),
                                             shuffle=True,
                                             batch_size= 10,
                                            class_mode="binary",
                                             subset = 'training')


# In[93]:


test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load images directly from the test directory
test_data = test_gen.flow_from_directory(
    directory=test_dir,        # The main test directory containing subfolders for each class
    target_size=(224, 224),    # Resize images to match model input
    batch_size=10,             # Adjust batch size as needed
    shuffle=False,             # No shuffling for test data
    class_mode="binary",           # No labels needed for inference
)


# In[94]:


val_dir = r"C:\Users\vansh\Downloads\chest_xray\val"

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_data = train_gen.flow_from_directory(directory=train_dir,
                                             target_size = (224,224),
                                             shuffle=True,
                                             batch_size= 10,
                                             subset = 'validation')


# In[95]:


disease_cls = ['pneumonia','normal']

# Normal Images

normal_path = os.path.join(train_dir, disease_cls[1], '*')
normal_images = glob.glob(normal_path)

plt.figure(figsize=(12,10))

for i in enumerate(normal_images[0:3]):
    plt.subplot(1,3, i[0]+1)
    img = cv2.imread(i[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.tight_layout()
    plt.imshow(img)


# In[96]:


# pneumonia Images

pneumonia_path = os.path.join(train_dir, disease_cls[0], '*')
pneumonia_images = glob.glob(pneumonia_path)

plt.figure(figsize=(12,10))

for i in enumerate(pneumonia_images[0:3]):
    plt.subplot(1,3, i[0]+1)
    img = cv2.imread(i[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.tight_layout()
    plt.imshow(img)


# In[97]:


# Image preprocessing functions

def unsharp_mask(image,radius=5,amount=1.5):

    image = img_as_float(image) # ensuring float values for computations

    blurred_image = gaussian_filter(image, sigma=radius)

    mask = image - blurred_image # keep the edges created by the filter
    sharpened_image = image + mask * amount

    sharpened_image = np.clip(sharpened_image, float(0), float(1)) # Interval [0.0, 1.0]
    sharpened_image = (sharpened_image*255).astype(np.uint8) # Interval [0,255]

    return sharpened_image #sharpened images

def clahe(image):
    x = exposure.equalize_adapthist(image ,clip_limit=0.08)
    return x


# In[98]:


# applying preprocessing on Normal Images

plt.figure(figsize=(12,10))

for i in enumerate(normal_images[0:3]):
    plt.subplot(1,3, i[0]+1)
    img = cv2.imread(i[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = unsharp_mask(img)
    img = clahe(img)
    plt.tight_layout()
    plt.imshow(img)


# In[99]:


# applying preprocessin on pneumonia images

plt.figure(figsize=(12,10))

for i in enumerate(pneumonia_images[0:3]):
    plt.subplot(1,3, i[0]+1)
    img = cv2.imread(i[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = unsharp_mask(img)
    img = clahe(img)
    plt.tight_layout()
    plt.imshow(img)


# In[61]:


# model traning (using "imagenet" a pretrained data model )
base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# In[ ]:


#history = model.fit(training_data, epochs=10, validation_data=validation_data)


# In[ ]:


# compiling the model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=["accuracy"])


# In[107]:


history = model.fit(
    training_data,
    epochs=7,     
    validation_data=validation_data  
)


# In[108]:


model.evaluate(test_data)


# In[109]:


model.save("Downloads/pneumonia_model1.keras") 


# In[ ]:




