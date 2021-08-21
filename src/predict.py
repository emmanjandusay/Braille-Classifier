import numpy as np
import cv2 as cv
import os.path
import matplotlib.pyplot as plt

from preproc import process_image
from generateModel import getModel, generatePickle, updateModel
def predictCell(image):
    categories = ['binata', 'buhay', 'dalaga', 'eksamen', 'ewan', 'gunita', 'halaman', 'hapon', 'isip', 'kailangan',
             'karaniwan', 'kislap', 'larawan', 'mabuti', 'noon', 'opo', 'papaano', 'patuloy', 'roon', 'subalit',
              'talaga', 'ugali', 'wasto']

    image = process_image(image)
    image = np.expand_dims(image, 0)

    if os.path.isfile('braille-model.pickle'):
        print('Model found...')
        model = getModel()

        print('Predicting captured cell...')

        prediction = model.predict(image)

        predInt = prediction[0]
        predictedClass = categories[prediction[0]]

        print('Prediction Integer is :', predInt)
        print('Prediction is :', predictedClass)

        return predictedClass

    else:
        print('Model not found...')
        generatePickle()
        model = getModel()

        print('Predicting captured cell...')

        prediction = model.predict(image)

        predInt = prediction[0]
        predictedClass = categories[prediction[0]]

        print('Prediction Integer is :', predInt)
        print('Prediction is :', predictedClass)

        return predictedClass