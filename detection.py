import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input,decode_predictions
from keras.preprocessing.image import  load_img
from keras_preprocessing.image import img_to_array

model = ResNet50(weights ='imagenet')

image = load_img('strawberry.jpg', targest_size= (224,224)) #resizing done
image = img_to_array(image)


def detection(model, image):
    image = np.expand_dims(image , axis = 0) #extra axis?
    image_processed = preprocess_input(image)
    ymod =