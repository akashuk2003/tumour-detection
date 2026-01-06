import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64
import os
from django.conf import settings

def process_and_segment(image_path):
    """
    Reads an MRI scan, segments the bright tumor area, 
    calculates pixel area, and saves the result.
    """
    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None:
        return None, 0
    
    # 2. Convert to Grayscale & Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Thresholding (finding bright spots/tumors)
    # Adjust 45, 255 based on your MRI brightness
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

    # 4. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Calculate Area & Draw Contours
    tumour_area = 0
    # Draw green outlines on the original image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    for c in contours:
        tumour_area += cv2.contourArea(c)

    # 6. Save Processed Image
    # We save it to the same directory with a suffix
    directory, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_processed{ext}"
    save_path = os.path.join(directory, new_filename)
    
    cv2.imwrite(save_path, img)

    # Return just the filename (relative to media root) for the template
    return new_filename, tumour_area

def generate_comparison_chart(area1, area2):
    """
    Generates a bar chart comparing two tumor areas 
    and returns it as a base64 string.
    """
    plt.switch_backend('AGG') # Non-GUI backend
    plt.figure(figsize=(6, 4))
    
    # Data
    labels = ['Scan 1', 'Scan 2']
    values = [area1, area2]
    colors = ['#3498db', '#e74c3c'] # Blue and Red

    # Plot
    plt.bar(labels, values, color=colors)
    plt.title('Tumour Growth Comparison')
    plt.ylabel('Area (pixels)')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    
    plt.close()
    return uri