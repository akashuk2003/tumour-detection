#  Brain Tumor Detection & Analysis System

A web-based AI application that detects brain tumors from MRI scans, visualizes the tumor area using computer vision, and tracks tumor growth over time.

##  Features

* **Tumor Detection:** Uses a Deep Learning model (MobileNetV2) to classify MRI scans as "Tumor" or "Healthy".
* **AI Segmentation:** Uses OpenCV to highlight the tumor area and calculate its size in pixels.
* **Growth Comparison:** Compares two different scans (e.g., from different dates) to calculate tumor growth or shrinkage percentage.
* **Visual Reports:** Generates bar charts and side-by-side comparisons.

##  Tech Stack

* **Backend:** Django (Python)
* **AI/Deep Learning:** TensorFlow, Keras, MobileNetV2 (Transfer Learning)
* **Computer Vision:** OpenCV (cv2)
* **Visualization:** Matplotlib
* **Frontend:** HTML/CSS

Installation & Setup
Clone the Repository

Bash

cd tumour_dl_system
Create a Virtual Environment

Bash

python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
Install Dependencies

Bash

pip install -r requirements.txt
Database Setup

Bash

python manage.py makemigrations
python manage.py migrate
Train the AI Model (Only needed if cnn_model.h5 is missing or you want to retrain)

Bash

python detector/dl/train_model.py
Run the Server

Bash

python manage.py runserver
Open your browser at http://127.0.0.1:8000/.

🖥️ How to Use
Single Analysis:

Go to the home page.

Upload an MRI image (JPG/PNG).

View the prediction (Tumor/No Tumor) and the green segmented area.

Growth Comparison:

Click "Compare Growth" in the navigation.

Upload an "Older Scan" and a "Newer Scan".

The system will calculate the pixel area difference and show if the tumor has grown or shrunk.