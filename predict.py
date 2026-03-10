import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load model
model = load_model("emotion_model.h5")

# read image using full path
img = cv2.imread(r"C:\Users\lavan\Downloads\Autism_CNN_Project\test.jpg")

if img is None:
    print("Image not found! Check file path.")
    exit()

# resize
img = cv2.resize(img,(48,48))

# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# normalize
img = img/255.0

# reshape
img = img.reshape(1,48,48,1)

# predict
prediction = model.predict(img)

labels = ["anger","fear","joy","natural","sadness","surprise"]

print("Prediction:", labels[np.argmax(prediction)])