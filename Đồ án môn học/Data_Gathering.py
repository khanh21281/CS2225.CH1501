# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:36:43 2020

@author: nvtda
"""

import cv2
import os
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
#size = 4
face_detector = cv2.CascadeClassifier('C:\\Users\\nvtda\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1, 1) # flip video image vertically
    
    mini = cv2.resize(img, (img.shape[1] , img.shape[0]))
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(mini)    
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("D:\ML projects\data_gathering\\user." + str(face_id) + '.' +  
                    str(count) + ".jpg", mini[y:y+h,x:x+w])
        cv2.imshow('image', img)
        
    k = cv2.waitKey(300) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 120: # Take n face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()