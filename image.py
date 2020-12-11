from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2 
import os
import numpy
## to suppress tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

## load model
model = tf.keras.models.load_model('model/model.h5',compile = False)


def detect_points(face_img):
    me  = np.array(face_img)
    x_test = np.expand_dims(me, axis=0)
    x_test = np.expand_dims(x_test, axis=3)

    y_test = model.predict(x_test)
    
    return y_test
    
# Load haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dimensions = (96, 96)

# Enter the path to your test image
img = cv2.imread('inputs/face4.jpg')
default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
faces_img = np.copy(gray_img)
plt.rcParams["axes.grid"] = False

## No face detected
if len(faces) == 0:
    print("No faces detected")

for i, (x,y,w,h) in enumerate(faces):
    
    ## isolate face to pass to the model
    just_face = cv2.resize(gray_img[y:y+h,x:x+w], dimensions)
    just_color_face = cv2.resize(img[y:y+h,x:x+w], dimensions)
    
    ## Draw a rectange to debug
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    
    label_point = detect_points(just_face)
    ## go through each of the points and draw a circle in the img
    for i in range(0,30):
        if i%2 == 0:
         cv2.circle(just_color_face,(int(label_point[0][i]),int(label_point[0][i+1])),1,(255,0,0),1)
    
    ## resize the face to original dimension and put it back in the picture 
    just_color_face = cv2.resize(just_color_face,(h,w))
    img[y:y+h,x:x+w] = just_color_face
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    ## save the final output
    cv2.imwrite("output/output.jpg",img)
    ## show the final output
    plt.imshow(img)
    plt.show()