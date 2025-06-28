# detection/views.py

import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from .forms import ImageUploadForm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from django.http import HttpResponse
from PIL import Image
import io
from keras.models import load_model
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, 'cyst_detection_model.h5')
model = load_model(model_path)

# Define the Gaussian kernel
kernel_size = (4, 4)
sigma_x = 1.0
sigma_y = 1.0
kernel_x = cv2.getGaussianKernel(kernel_size[0], sigma_x)
kernel_y = cv2.getGaussianKernel(kernel_size[1], sigma_y)
gaussian_kernel = np.outer(kernel_x, kernel_y)

def detect_cysts(image_path, model, gaussian_kernel):
    # Read the image from the provided path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at path {image_path}")
        return
    
    # Resize the image to the input size expected by the model
    sample_image = cv2.resize(image, (128, 128))
    
    # Prepare the image for model prediction
    sample_image = np.expand_dims(sample_image, axis=0)
    sample_image = sample_image / 255.0
    
    # Predict with the model
    prediction = model.predict(sample_image)
    predicted_class = "Infected" if prediction > 0.5 else "Not Infected"
    print("Predicted class:", predicted_class)

    if predicted_class == "Infected":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eroded_image = cv2.erode(gray_image, gaussian_kernel, iterations=1)
        _, thresholded_image = cv2.threshold(eroded_image, 20, 255, cv2.THRESH_BINARY)
        inverse_thresholded_image = 255 - thresholded_image

        contours, _ = cv2.findContours(inverse_thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of contours detected:", len(contours))

        # Draw contours on the original image
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

        # Display the result
        plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        plt.title('Infected Image with Cysts Contours')
        plt.axis('off')
        plt.show()

    else:
        print("Not Infected")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eroded_image = cv2.erode(gray_image, gaussian_kernel, iterations=1)
        _, thresholded_image = cv2.threshold(eroded_image, 20, 255, cv2.THRESH_BINARY)
        inverse_thresholded_image = 255 - thresholded_image

        # Display the result
        plt.imshow(cv2.cvtColor(inverse_thresholded_image, cv2.COLOR_GRAY2RGB))
        plt.title('Image without cysts')
        plt.axis('off')
        plt.show()

# views.py

def upload_and_detect(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            image_file = request.FILES['image']
            image_path = default_storage.save('temp_image.jpg', image_file)
            image_full_path = default_storage.path(image_path)

            # Ensure the image path is correct
            if not os.path.exists(image_full_path):
                return render(request, 'upload.html', {'form': form, 'error': 'File not found'})

            # Call the detect_cysts function
            detect_cysts(image_full_path, model, gaussian_kernel)
            
            return render(request, 'result.html', {'result': 'Detection complete'})
    else:
        form = ImageUploadForm()
    
    return render(request, 'upload.html', {'form': form})

