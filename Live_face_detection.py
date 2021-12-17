'''
objective: detect Face from live camera and draw rectangle around face
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

# video capture from webcam
video = cv2.VideoCapture(0)
'''
    load face haar cascade
    HarrCascadeClassifier is a class that represents a classifier trained to detect faces.
    Refrence: https://github.com/opencv/opencv/tree/master/data/haarcascades
    
    Note: you can change harr cascade file to detect other objects
'''
face_cascadde   = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# start looping
while True:
    check, image = video.read()
    if check:
    # conver to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # load face cascade 
        # You can change the scaleFactor and minNeighbors to increase/decrease the accuracy but it will increase the time it takes to detect the face, it is recommended to keep the default value between 1.05 and 3 and 5
        
        face = face_cascadde.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5)

        # draw rectangle around face
        for x, y, w, h in face:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            # wirite message
            cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 200), 2)

        # display image
        cv2.imshow('Original', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()
cv2.release()