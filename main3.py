from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import time
import cv2 
import os
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
#from keras.preprocessing import image
# define a video capture object 
model = tf.keras.models.load_model('model.h5',compile = False)
print(model.summary())


def detect_points(face_img):
    me  = np.array(face_img)
    x_test = np.expand_dims(me, axis=0)
    x_test = np.expand_dims(x_test, axis=3)

    y_test = model.predict(x_test)
    #label_points = (np.squeeze(y_test)*48)+48 
    
    return y_test
    
# Load haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dimensions = (96, 96)

# Enter the path to your test image
vid = cv2.VideoCapture(0)
while True:
    ret,frame = vid.read()
    img = frame
    default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    #faces = face_cascade.detectMultiScale(gray_img, 4, 6)
    #print("Number of faces:" , len(faces))
    faces_img = np.copy(gray_img)

    plt.rcParams["axes.grid"] = False


    all_x_cords = []
    all_y_cords = []
    if len(faces) == 0:
        print("No faces found!")
    for i, (x,y,w,h) in enumerate(faces):
        print(x,y,w,h)
        #h += 10
        #w += 10
        #x -= 5
        #y -= 5

        just_face = cv2.resize(gray_img[y:y+h,x:x+w], dimensions)
        just_color_face = cv2.resize(img[y:y+h,x:x+w], dimensions)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        #plt.imshow(just_color_face)
        #plt.show()
        scale_val_x = w/96
        scale_val_y = h/96

        label_point = detect_points(just_face)
        for i in range(0,30):
            if i%2 == 0:
             #print(i)
             #print(label_point[0][i])
             cv2.circle(just_color_face,(int(label_point[0][i]),int(label_point[0][i+1])),1,(255,0,0),1)

        just_color_face = cv2.resize(just_color_face,(480,480))
        #plt.imshow(just_color_face)
        #plt.show()
        # Display the resulting frame 
        cv2.imshow('frame', just_color_face) 
      
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break



    #all_x_cords.append((label_point[::2]*scale_val_x)+x)
    #all_y_cords.append((label_point[1::2]*scale_val_y)+y)
   
   
    #plt.imshow(faces_img, cmap='gray')
    #plt.plot(label_point[::2], label_point[1::2], 'ro', markersize=5)
    #plt.show()
    
    
#plt.imshow(default_img)    
#plt.plot(all_x_cords, all_y_cords, 'wo',  markersize=3)
#plt.show()