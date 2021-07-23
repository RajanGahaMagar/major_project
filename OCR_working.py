import numpy as np
import cv2
import pickle
import pytesseract


#######################################
width = 640
height = 480
threshold = 0.9
#######################################

cap = cv2.VideoCapture()
cap.set(3,width)
cap.set(4,height)

pickle_in = open("trained_(0-9)Fnt.p","rb")
model = pickle.load(pickle_in)

def preProcessing(img, j):
    img = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret,img = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #img = cv2.equalizeHist(img)
    #img = img/255
    #cv2.imwrite(f'images/{j}.jpg', img)
    return img

video = cv2.VideoCapture(0)
j = 0
while True:
    j = j+1
    ret, image = video.read()
    original = image
    original = preProcessing(original, j)
    height, width = original.shape[0], original.shape[1]

    d = pytesseract.image_to_boxes(image, output_type=pytesseract.Output.DICT)

    for i in range(len(d['char'])):
        try:
            (text, x1, y2, x2, y1) = (d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])
            img = cv2.rectangle(original, (x1, height - y1), (x2, height - y2), (0, 255, 0), 2)
            cv2.imshow('original', img)
            # pass this character into your model
            character = original[(height - y2):(height - y1), x1:x2]/255
            character = cv2.resize(character, (32, 32))
            cv2.imshow("Processed Image", character)
            character = character.reshape(1, 32, 32, 1)

            classIndex = int(model.predict_classes(character))
            predictions = model.predict(character)
            probVal = np.amax(predictions)
            # Predict
            if probVal > threshold:
                print(classIndex, end=" ")
                cv2.putText(image, str(classIndex) + "   " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255), 1)
        except:
            pass
    print("\n")



    #cv2.imshow("Original Image",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
