from django.shortcuts import render
from django.conf import settings
import os

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
        os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists
        image_path = os.path.join(upload_dir, uploaded_image.name)

        # Save the image
        with open(image_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)

        # Prepare the path to the image to pass to the template
        display_img = os.path.join(settings.MEDIA_URL, 'uploads', uploaded_image.name)

    return render(request, 'detect.html', {'display_img': display_img})
