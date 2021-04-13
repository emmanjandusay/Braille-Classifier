import cv2
import imageio
import time
import tkinter as tk, threading
import numpy as np

from predict import predictCell
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS, 30)
scale=1
capture_frame = np.zeros((1280,720,3), np.uint8)
isClicked = False

def stream(label, predictLabel):
    while(True):
        global capture_frame, isClicked
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = resizeImg(frame)
        frame = checkZoom(frame)
        frame = drawMarker(frame)
        
        frame_image = ImageTk.PhotoImage(Image.fromarray(frame))
        label.config(image=frame_image)
        label.image = frame_image
        
        if isClicked:
            img = getroi(frame)
            text = "Predicting captured region..."
            predictLabel.set(text)
            predClass = predictCell(img)
            text = "Predicted class: " + predClass
            predictLabel.set(text)
            isClicked = False

def checkZoom(frame):
    global scale
    height, width, channels = frame.shape
    #prepare the crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(centerX*scale),int(centerY*scale)
    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height)) 
    return resized_cropped

def drawMarker(frame):
    ul, br = getCoord(frame)
    cv2.rectangle(frame, ul, br, (0, 255, 0), thickness=1)
    return frame

def resizeImg(image):
    width = 475
    height = 250
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def getroi(image):
    ul, br = getCoord(image)
    x_u, y_u = ul #Get upper left coordinates
    x_b, y_b = br #Get lower right point coordinates

    cell = image[y_u:y_b, x_u:x_b]

    return cell

def getCoord(image):
    #Calculate center of image
    h, w, ch = image.shape

    x = w // 2
    y = h // 2

    upper_left = (x - (w // 8), y - (h // 6))
    bottom_right = (w * 2 // 3, h * 2 // 3)

    return upper_left, bottom_right

def addScale():
    global scale
    if scale == 1:
        pass
    else:
        scale += 0.10

def subScale():
    global scale
    if scale == 0.5:
        pass
    else:
        scale -= 0.10

def predictModel():
    global isClicked
    isClicked = True

if __name__ == "__main__":

    root = tk.Tk()
    root.title("Two-Cell Filipino Brailler Classifier")

    app_width = 480
    app_height = 320

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (app_width / 2)
    y = (screen_height / 2) - (app_height / 2)

    root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')

    my_label = tk.Label(root)
    my_label.grid(row=0, column=0, columnspan=5)

    button_predict = tk.Button(root, text="Predict", command=predictModel)
    button_predict.grid(row=2, column=0)

    predText = tk.StringVar()
    predText.set("On Standby")
    predict_label = tk.Label(root, textvariable=predText)
    predict_label.grid(row=1,column=0, columnspan=5)

    button_sub = tk.Button(root, text="-", command=addScale)
    button_sub.grid(row=2, column=1)

    zoom_label = tk.Label(root, text="Zoom")
    zoom_label.grid(row=2, column=2)

    button_add = tk.Button(root, text="+", command=subScale)
    button_add.grid(row=2, column=3)

    button_exit = tk.Button(root, text="Exit", command=root.quit)
    button_exit.grid(row=2, column=4)

    thread = threading.Thread(target=stream, args=(my_label, predText,))
    thread.daemon = 1
    thread.start()
    root.mainloop()