import matplotlib.pyplot as plt
import pytesseract
import cv2

original = cv2.imread('C:/Users/Ranjeet Gupta/Desktop/Final/images/23.jpg')
img = original

height, width = img.shape[0], img.shape[1]
d = pytesseract.image_to_boxes(img, output_type=pytesseract.Output.DICT)
print("d=",d)
n_boxes = len(d['char'])
for i in range(n_boxes):
    
    #get segmented co-ordinates
    (text,x1,y2,x2,y1) = (d['char'][i],d['left'][i],d['top'][i],d['right'][i],d['bottom'][i])
    
    #extract individual characters
    try:
        crop = original[(height-y2):(height-y1), x1:x2]
    except:
        pass
    cv2.imwrite(f"character/crop_{i}.png".format(i), crop)
    
    #overlay rectangles to visualize
    #comment this line in actual code
    img = cv2.rectangle(img, (x1,height-y1), (x2,height-y2) , (0,255,0), 2)
    
fig, ax = plt.subplots(figsize=(20, 4))
ax.imshow(original)
cv2.waitKey(0)