import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input,decode_predictions
from keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

model = ResNet50(weights ='imagenet')

image = load_img('strawberry.jpg', target_size=(224,224))
image = img_to_array(image)


def detection(model, image):
    image = np.expand_dims(image , axis = 0) #extra axis?
    image_processed = preprocess_input(image)
    predictions = model.predict(image_processed)
    translated = decode_predictions(predictions)
    translated = translated[0][0]
    return translated

print('%s (%.2f%%)' % (detection(model, image)[1], detection(model,image)[2]*100))
