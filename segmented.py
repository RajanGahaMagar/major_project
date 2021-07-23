import matplotlib.pyplot as plt
import pytesseract
import cv2

img = cv2.imread('image2.jpg')
height, width = img.shape[0], img.shape[1]


d = pytesseract.image_to_boxes(img, output_type=pytesseract.Output.DICT)

for i in range(len(d['char'])):
    (text,x1,y2,x2,y1) = (d['char'][i],d['left'][i],d['top'][i],d['right'][i],d['bottom'][i])
    
    #pass this character into your model
    character = original[(height-y2):(height-y1), x1:x2]
    #resize the character using cv2.resize() if your model takes fixed image size