# This code is used to test the model.

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import matplotlib.pyplot as plt
import cv2


class CustomConv2DTranspose(Conv2DTranspose):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(filters, kernel_size, strides=strides, padding=padding, output_padding=output_padding, **kwargs)


get_custom_objects().update({'CustomConv2DTranspose': CustomConv2DTranspose})

path = 'UNet_model_compatible.h5'

try:
    model = load_model(path)
    print("Model loaded successfully.")

except Exception as e:
    print("Error loading model:", e)

#Loding gray scale image.
test_path=cv2.imread('D:\\New folder\\lungs\\lung_109.png', cv2.IMREAD_GRAYSCALE)
prediction_path=cv2.imread('D:\\New folder\\mask\\lung_109.png', cv2.IMREAD_GRAYSCALE)

#512*512
print("Image Shape:", test_path.shape) 
print("Image Shape:", prediction_path.shape)

#processing of input as per the model input.
testX = test_path.astype(np.float32)
testY = prediction_path.astype(np.float32)

testX = (testX - 127.0) / 127.0 
testX = np.reshape(testX, (1, 512, 512, 1))
 
testY = (testY>127).astype(np.float32)
testY = np.reshape(testY, (1, 512, 512, 1))

#function to display output
def display_result(original_img, prediction, overlay_img, actual_prediction, actual_overlay_img):
    plt.figure(figsize=(8, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img.squeeze(), cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title("Segmentation Output from test")
    plt.imshow(prediction.squeeze(), cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Overlay from test")
    plt.imshow(overlay_img.squeeze(), cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("Original Image")
    plt.imshow(original_img.squeeze(), cmap='gray')
    
    plt.subplot(2, 3, 5)
    plt.title("actual Segmentation output")
    plt.imshow(actual_prediction.squeeze(), cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("actual Overlay")
    plt.imshow(actual_overlay_img.squeeze(), cmap='gray')

    plt.show()

input_img = testX
prediction = model.predict(input_img)
print(prediction.shape) #(1, 512, 512, 1)

overlay = cv2.addWeighted(np.squeeze(testX), 0.5, np.squeeze(prediction), 0.5, 0)
actual_overlay = cv2.addWeighted(np.squeeze(testX), 0.5, np.squeeze(testY), 0.5, 0)

display_result(testX, prediction, overlay, testY, actual_overlay)

    

