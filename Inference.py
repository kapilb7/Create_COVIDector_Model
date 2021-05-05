#!/usr/bin/env python3

import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import PIL
from PIL import Image
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="/Users/kapil/Documents/FYP/COVID Models/Final/COVIDNetV2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
"""
print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])
"""

class_names = ['negative', 'positive']

# Read the image and decode to a tensor
image_path='/Users/kapil/Downloads/3207414e-9ff0-4d36-a029-9daed39786b1.jpg' 
img = cv2.imread(image_path)
img = cv2.resize(img,(224,224))
img = np.array(img, dtype=np.float32)
img.shape = (1, 224, 224, 3)
#Preprocess the image to required size and cast
#print(img.dtype)

interpreter.set_tensor(input_details[0]['index'], img)
# Run inference
interpreter.invoke()
# Get prediction results
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
score = tf.nn.softmax(tflite_model_predictions[0])
#print("Prediction results shape:", tflite_model_predictions)
print(
	"This image most likely belongs to {} with a {:.2f} percent confidence."
	.format(class_names[np.argmax(score)], 100 * np.max(score))
)

