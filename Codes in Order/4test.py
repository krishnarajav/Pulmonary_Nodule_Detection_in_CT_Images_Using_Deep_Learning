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
path = "D:\\Zmajorproject\\imp\\codes\\UNet_model_compatible.h5"

try:
    model = load_model(path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

image_path = "D:\\Zmajorproject\\imp\\Test Images\\lungs\\lung_1.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
temp = img

testX=img = img.astype(np.float32)

testX = (testX - 127.0) / 127.0 
testX = np.reshape(testX, (1, 512, 512, 1))

def display_result(original_img, prediction, overlay_img=None):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img.squeeze(), cmap='bone')
    
    plt.subplot(1, 3, 2)
    plt.title("Segmentation Output")
    plt.imshow(prediction.squeeze(), cmap='gray')

    if overlay_img is not None:
        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(overlay_img.squeeze(), cmap='gray')

    plt.show()

prediction = model.predict(testX)

prediction_squeezed = np.squeeze(prediction)
prediction_squeezed = (prediction_squeezed>0.5).astype(np.int8)

temp = (temp * 0.5).astype(np.uint8)
overlay = np.where(prediction_squeezed == 1, 255, temp)
display_result(testX, prediction, overlay)
