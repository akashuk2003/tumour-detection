import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
# Load model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cnn_model.h5')
model = None

def load_dl_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        # Note: If your .h5 file was saved with an older Keras version, 
        # you might see warnings, but it usually works.
        model = tf.keras.models.load_model(MODEL_PATH)

def classify_image(img_path):
    load_dl_model()
    if model is None:
        return False, 0.0

    # 1. Load the image
    # FIX: Called directly without 'image.' prefix
    img = load_img(img_path, target_size=(224, 224))    
    # 2. Convert to array
    # FIX: Called directly without 'image.' prefix
    img_array = img_to_array(img)
    
    # 3. Create a batch (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predict
    prediction = model.predict(img_array)
    
    confidence = prediction[0][0]
    is_tumour = confidence > 0.5
    
    return is_tumour, float(confidence)