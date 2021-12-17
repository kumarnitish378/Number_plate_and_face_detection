'''
objective: detect live number plate and face from camera
Date: 2021-12-16
Author: Nitish Kumar Sharma
Email: nitish.ns378@gmail.com

License: MIT
MIT License (c) 2021 Nitish Kumar Sharma
This code is licensed under the MIT license (see LICENSE.txt for details)
You are free to use this code in your own projects, as long as you give credit to the original author.
all rights reserved. by Nitish Kumar Sharma

Note: This code is not optimized for performance. only for learning purpose. and it is not tested for performance.
    you may get some error in the code. if any error please let me know. 
    Telegram No: 7631256855
    whatsapp No: 7631256855s

'''

# import open cv and read image nad display it
from typing import Pattern
import cv2
import numpy as np
import imutils
# import easyocr
import sys
import os
# from PIL import Image
from pytesseract import pytesseract
import re
import copy

# Chnage path to your path
path_to_tesseract = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract


# load face haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def filter_text(text):
    '''
    Remove unwanted characters from text
    '''
    print("Filtering text")
    print("License plate Number (without Filter):{}".format(text.replace("\n\n",'')))
    Pattern = re.compile(r'[^a-zA-Z0-9\s]')
    text = Pattern.sub('', copy.copy(text))
    print("License plate Number:{}".format(text.replace("\n\n",'')))


video = cv2.VideoCapture(0)
while True:
    _, image = video.read()
    cv2.imshow("Capturing", image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ---------Fcae Detection-----------------
    face = face_cascade.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5)

    # draw rectangle around face
    for x, y, w, h in face:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # wirite message
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 200), 2)
    
    
    # ---------Number Plate Detection-----------------
    # bilateral filter
    image_gray_blur = cv2.bilateralFilter(image_gray, 10, 75, 75)
    # cv2.imshow('blur', image_gray_blur)
    print("------------------Getting text from image--------------")
    results = pytesseract.image_to_string(image_gray_blur, lang='eng')
    filter_text(results)
    print("------------------Getting text from image--------------")
    edges = cv2.Canny(image_gray_blur, 100, 200)

    # find contours and apy a mask
    keypoint = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours = imutils.grab_contours(keypoint)
    countours = sorted(countours, key=cv2.contourArea, reverse=True)[:10]
    # print(countours)
    for c in countours:
        # compute the center of the contour
        # approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        approx = cv2.approxPolyDP(c,10, True)
        if len(approx) == 4:
            ll = approx
            location = cv2.minAreaRect(c)
            box = cv2.boxPoints(location)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
    cv2.imshow('rectangles', image)
    # masking the image
    try:
        mask = np.zeros(image_gray.shape, dtype="uint8")
        new_img = cv2.drawContours(mask, [ll], 0, 255, -1)
        new_img = cv2.bitwise_and(image, image, mask=new_img)

        (x,y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        crop_img = image[x1:x2+1, y1:y2+1]
        print("Getting text from image")
        results = pytesseract.image_to_string(crop_img, lang='eng')
        filter_text(results)
        print("Done")
    except:
        print("Error")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.release()
cv2.destroyAllWindows()
