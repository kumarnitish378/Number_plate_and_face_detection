'''
objective: detect face from image and draw rectangle around face

Date: 2021-12-16
Author: Nitish Kumar Sharma
Email: nitish.ns378@gmail.com

License: MIT
MIT License (c) 2021 Nitish Kumar Sharma
This code is licensed under the MIT license (see LICENSE.txt for details)
You are free to use this code in your own projects, as long as you give credit to the original author.
all rights reserved. by Nitish Kumar Sharma

'''


import cv2

# read image from images folder and display it
image = cv2.imread('images\\f3.jpeg')

'''
    load face haar cascade
    HarrCascadeClassifier is a class that represents a classifier trained to detect faces.
    Refrence: https://github.com/opencv/opencv/tree/master/data/haarcascades
    
    Note: you can change harr cascade file to detect other objects
'''
face_cascade   = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# conver to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load face cascade
face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5)

# draw rectangle around face
for x, y, w, h in face:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # wirite message
    cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 200), 2)

# display image
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()