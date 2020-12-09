import cv2 
import os
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
# define a video capture object 
vid = cv2.VideoCapture(0) 
model = tf.keras.models.load_model('model.h5',compile = False)
print(model.summary())

def get_coordinates(arr, scale = 96):
    x, y = [], []
    for i in range(len(arr)):
        if i % 2 == 0:
            x.append(arr[i] * scale)
        else:
            y.append(arr[i]* scale)
    return x, y
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    frame = cv2.resize(frame,(96,96))
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x = numpy.reshape(x,(1,96,96,1))
    #print(x.shape)
    points = model.predict(x)
    arr = get_coordinates(points)
    print(arr)
    for i in range(0,30):
        if i%2 == 0:
            cv2.circle(frame, (arr[0][0][i],arr[0][0][i+1]) , 10 , (255,0,0),2)

    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
