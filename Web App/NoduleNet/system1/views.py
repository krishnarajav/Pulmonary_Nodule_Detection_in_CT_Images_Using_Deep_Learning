from django.shortcuts import render
from django.conf import settings
import os

# Model imports.
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import matplotlib.pyplot as plt
import cv2

from datetime import datetime
from skimage.measure import label, regionprops

class CustomConv2DTranspose(Conv2DTranspose):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(filters, kernel_size, strides=strides, padding=padding, output_padding=output_padding, **kwargs)


filename=None
#model metadata
model_version = "1.0"
model_status = "Ready"
model_accuracy = "82.6%"
last_updated = "21-11-2024"

def process(request):

    if filename == None:
        timestamp = datetime.now().timestamp()
        return render(request, 'detect.html', {'display_img': None, 'timestamp':timestamp})
    
    get_custom_objects().update({'CustomConv2DTranspose': CustomConv2DTranspose})
    path = 'UNet_model_compatible.h5'

    try:
        model = load_model(path)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)

    image_path=os.path.join(settings.MEDIA_ROOT, 'uploads', 'input.png')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    temp=img # Stored to display at the end as output.
    testX=img = img.astype(np.float32)

    if img is None:
        print("Error: Unable to load image.")
        return
        
    testX = (testX - 127.0) / 127.0 
    testX = np.reshape(testX, (1, 512, 512, 1))

    prediction = model.predict(testX)

    prediction_squeezed = np.squeeze(prediction)
    prediction_squeezed = (prediction_squeezed>0.5).astype(np.int8)

    # Find connected components (nodules)
    labeled_image = label(prediction_squeezed)
    regions = regionprops(labeled_image)

    # Initialize a list to store nodule details
    nodule_details = []

    for region in regions:
        # Bounding box: (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox

        # Calculate the diameter (approximate as the diagonal of the bounding box)
        diameter = np.sqrt((max_row - min_row) ** 2 + (max_col - min_col) ** 2)

        # Calculate the centroid (location) of the nodule
        centroid = region.centroid  # (row, col)

        # Store the nodule details
        nodule_details.append({
            'nodule_number': len(nodule_details) + 1,
            'diameter': round(diameter, 3),
            'centroid': centroid,
            'bbox': (min_row, min_col, max_row, max_col)
        })

    temp = (temp * 0.5).astype(np.uint8)

    overlay = np.where(prediction_squeezed == 1, 255, temp)

    output_filename = os.path.join(settings.MEDIA_ROOT, 'uploads', 'output.png')
    cv2.imwrite(output_filename, overlay)
    display_img=os.path.join(settings.MEDIA_URL, 'uploads', 'output.png')

    timestamp = datetime.now().timestamp()
    model_analysis = "Completed"
    context = {
        'display_img': display_img, 
        'timestamp': timestamp,
        'version' : model_version,
        'status' : model_status,
        'accuracy' : model_accuracy,
        'last_updated' : last_updated,
        'analysis' : model_analysis, 
        'nodule_details': nodule_details, 
        'nodule_count':len(nodule_details)
    }
    
    return render(request, 'detect.html', context)

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def detect(request):
    display_img = None

    if request.method == "POST" and request.FILES.get('image'):
        # Get the uploaded image
        uploaded_image = request.FILES['image']

        # Define the path where you want to save the image
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        image_path = os.path.join(upload_dir, "input.png")
        global filename
        filename = image_path

        # Save the image
        with open(image_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)
        
        display_img = os.path.join(settings.MEDIA_URL, 'uploads', "input.png")

    timestamp = datetime.now().timestamp()
    
    if display_img is None:
        model_analysis = "File Not Uploaded"
    else:
        model_analysis = "Pending"
    
    context = {
        'display_img': display_img, 
        'timestamp': timestamp,
        'version' : model_version,
        'status' : model_status,
        'accuracy' : model_accuracy,
        'last_updated' : last_updated,
        'analysis' : model_analysis
    }
    return render(request, 'detect.html', context)
