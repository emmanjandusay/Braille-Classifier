import cv2 as cv
import os
import pickle
import random

from ELMR import ELMClassifier
from preproc import process_image
from sklearn.model_selection import train_test_split

def generatePickle():
    data = []
    categories = ['binata', 'buhay', 'dalaga', 'eksamen', 'ewan', 'gunita', 'halaman', 'hapon', 'isip', 'kailangan',
             'karaniwan', 'kislap', 'larawan', 'mabuti', 'noon', 'opo', 'papaano', 'patuloy', 'roon', 'subalit',
              'talaga', 'ugali', 'wasto']
    
    dir = 'C:\\Users\\Scadoodie\\Desktop\\Braille2C_Datasets'

    print('Generating pickle model...')

    for category in categories:
        path = os.path.join(dir,category)
        print("Current directory being pre-processed: ", path)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path,img)
            orig_img = cv.imread(imgpath,1) #Load image in Color

            try:
                image = process_image(orig_img)
                data.append([image,label])

            except Exception as e:
                pass
        
    if (len(data) <= 0):
        print("Data is empty")
    else:
        #Data length should contain 1180 images
        print("Success! braille-model.pickle generated.")
        print("Data Length: ", len(data))
        pick_in = open('braille-model.pickle','wb')
        pickle.dump(data,pick_in)
        pick_in.close()

def generateELM():
    features = []
    labels = []

    pick_in = open('braille-model.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    print('Generating ELM model...')

    random.shuffle(data)

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)
    model = ELMClassifier(n_hidden=500, activation_func='multiquadric')
    model.fit(xtrain, ytrain)

    print("Success! braille-ELM.sav generated.")
    pick = open('braille-ELM.sav','wb')
    pickle.dump(model,pick)
    pick.close()

def updateModel(data):
    random.shuffle(data)
    features = []
    labels = []

    pick_in = open('braille-model.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    print('Updating model...')
    
    random.shuffle(data)

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)
    model = ELMClassifier(n_hidden=500, activation_func='multiquadric')
    model.fit(xtrain, ytrain)

    pick = open('braille-ELM.sav','wb')
    pickle.dump(model,pick)
    model = pickle.load(pick)
    pick.close()

    return model, xtrain, xtest, ytrain, ytest
    
def getPickle():
    pick = open('braille-model.pickle','rb')
    model=pickle.load(pick)
    pick.close()
    return model

def generate():
    generatePickle()
    generateELM()