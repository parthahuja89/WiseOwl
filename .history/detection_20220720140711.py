"""
CopyRighted under CC license.
email parthahuj@gmail.com for usage.
"""
import numpy as np
import sys
from keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

import matplotlib.pyplot as plt
#model added
def detect(image_url):
    pass 
    model = ResNet50(weights='imagenet')
    image = load_img(image_url, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) #fixed the 3D errors
    image_processed = preprocess_input(image)
    predictions = model.predict(image_processed)
    translated = decode_predictions(predictions)
    translated = translated[0]
    return translated

def kill():
    K.clear_session()
   