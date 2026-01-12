import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
# Load model once - try new model first, then fallback to older versions
MODEL_PATH_NEW = os.path.join(os.path.dirname(__file__), 'cnn_model_new.h5')
MODEL_PATH_KERAS = os.path.join(os.path.dirname(__file__), 'cnn_model.keras')
MODEL_PATH_H5 = os.path.join(os.path.dirname(__file__), 'cnn_model.h5')
model = None

def load_dl_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH_NEW):
            model = tf.keras.models.load_model(MODEL_PATH_NEW)
        elif os.path.exists(MODEL_PATH_KERAS):
            model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        elif os.path.exists(MODEL_PATH_H5):
            model = tf.keras.models.load_model(MODEL_PATH_H5)

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
    
    # 3. Rescale to [0, 1] range to match training preprocessing
    img_array = img_array / 255.0
    
    # 4. Create a batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Predict
    prediction = model.predict(img_array)
    
    confidence = prediction[0][0]
    is_tumour = confidence > 0.5
    
    return is_tumour, float(confidence)