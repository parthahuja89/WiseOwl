"""
CopyRighted under CC license.
email parthahuj@gmail.com for usage.
"""
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

image = load_img('strawberry.jpg', target_size=(224, 224))
image = img_to_array(image)

#model added
def detection(model, image):
    image = np.expand_dims(image, axis=0) #fixed the 3D errors
    image_processed = preprocess_input(image)
    predictions = model.predict(image_processed)
    translated = decode_predictions(predictions)
    translated = translated[0]
    return translated



def plotting(image, predictions):

    plt.imshow(image)  #image predicted
    plt.figure()
    order = list(reversed(range(len(predictions))))


    xaxis = []
    for x in predictions:
        xaxis.append(x[2]*100)

    yaxis = []
    for y in predictions:
        yaxis.append(y[1])

    plt.barh(order, xaxis, alpha=0.5)
    plt.yticks(order, yaxis)
    plt.title('Precentage Predicted')
    plt.xlim(0, 100)

    plt.show()


if __name__ == "__main__":
    predictions = detection(model, image)
    plots = plotting(image, predictions)
