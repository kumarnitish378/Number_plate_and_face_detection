'''
objective: detect number plate from image
Date: 2021-12-16
Author: Nitish Kumar Sharma
Email: nitish.ns378@gmail.com

License: MIT
MIT License (c) 2021 Nitish Kumar Sharma
This code is licensed under the MIT license (see LICENSE.txt for details)
You are free to use this code in your own projects, as long as you give credit to the original author.
all rights reserved. by Nitish Kumar Sharma

'''

# Loading the required python modules
import pytesseract # this is tesseract module
import matplotlib.pyplot as plt
import cv2 # this is opencv module
import glob
import os


# specify path to the license plate images folder as shown below
path_for_license_plates = os.getcwd() + "/images/**/*.jpg"
list_license_plates = []
predicted_license_plates = []
  
for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
      
    license_plate_file = path_to_license_plate.split("/")[-1]
    license_plate, _ = os.path.splitext(license_plate_file)
    '''
    Here we append the actual license plate to a list
    '''
    list_license_plates.append(license_plate)
      
    '''
    Read each license plate image file using openCV
    '''
    img = cv2.imread(path_to_license_plate)
      
    '''
    We then pass each license plate image file
    to the Tesseract OCR engine using the Python library 
    wrapper for it. We get back predicted_result for 
    license plate. We append the predicted_result in a
    list and compare it with the original the license plate
    '''
    predicted_result = pytesseract.image_to_string(img, lang ='eng',
    config ='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
      
    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    predicted_license_plates.append(filter_predicted_result)