import serial
import pyttsx3
import time
import numpy as np
import cv2
import pickle
import pytesseract
import matplotlib.pyplot as plt


ser = serial.Serial('COM4',9600)  # open serial port
print("Connected to: "+ ser.name)         # check which port was really used

while(1):
    ser.flushInput()
    status = ser.read().decode()
    if status=='A':
        # Load Yolo using Deep neural network
        print("Object Detection Mode")
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)
        engine.say("Object Detection Mode Started")
        engine.runAndWait()
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        # Loading Classes Names
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Getting the name of three O/P layers of network along with their Indexes
        output_layers = net.getUnconnectedOutLayersNames()

        # Initializing the camera
        video = cv2.VideoCapture(0)
        while status!='B':
            # Loading image
            ret, frame = video.read()
            if not ret:
                print("Failed to Initialize Camera")
                break
            k = cv2.waitKey(1)
            if k % 256 == 27:
                break

            img = frame
            # height, width, channels = img.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        class_ids.append(class_id)
                        confidences.append(float(confidence))

            # Printing Detected Object Names
            for i in range(len(class_ids)):
                print(str(classes[class_ids[i]]) + " detected")
                engine = pyttsx3.init()
                engine.setProperty('rate', 125)
                engine.say(str(classes[class_ids[i]]) + " detected")
                engine.runAndWait()
                cv2.putText(frame, str(classes[class_ids[i]]) + "   " + str(confidences[i]), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255), 1)
            cv2.imshow("Detection Mode", frame)
            ser.flushInput()
            status = ser.read().decode()
            if status == 'B':
                video.release()
                cv2.destroyAllWindows()
                break

    ser.flushInput()
    status = ser.read().decode()
    status = 'B'
    if status=='B':
        print("Reading Mode")
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)
        engine.say("Reading Mode Started")
        engine.runAndWait()
        #######################################
        width = 640
        height = 480
        threshold = 0.6
        #######################################

        cap = cv2.VideoCapture()
        cap.set(3, width)
        cap.set(4, height)

        pickle_in = open("trained_model.p", "rb")
        model = pickle.load(pickle_in)


        def preProcessing(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255
            return img


        video = cv2.VideoCapture(0)
        while status!='A':
            ret, original = video.read()
            #original = cv2.imread('image4.jpg')
            image = original
            cv2.imshow("Processed Image", image)
            #img = cv2.resize(image, (32, 32))
            #img = preProcessing(img)
            # cv2.imshow("Processed Image",img)

            height, width = original.shape[0], original.shape[1]
            d = pytesseract.image_to_boxes(image, output_type=pytesseract.Output.DICT)
            for i in range(len(d['char'])):
                (text, x1, y2, x2, y1) = (d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])

                # pass this character into your model
                character = original[(height - y2):(height - y1), x1:x2]
                img = preProcessing(character)
                img = cv2.resize(img, (32, 32))
                img = img.reshape(1, 32, 32, 1)
                # Predict
                classIndex = int(model.predict_classes(img))
                # if classIndex>9 and classIndex<36:
                #     classIndex = str(chr(classIndex+87))
                predictions = model.predict(img)
                probVal = np.amax(predictions)

                if probVal > threshold:
                    print(str(classIndex))
            # if probVal > threshold:
            #     print(str(classIndex) + " detected")
            #     engine = pyttsx3.init()
            #     engine.setProperty('rate', 125)
            #     engine.say(str(classIndex) + " detected")
            #     engine.runAndWait()
            #     cv2.putText(image, str(classIndex) + "   " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
            #                 1, (0, 0, 255), 1)
            #
            # cv2.imshow("Reading Mode", image)
            ser.flushInput()
            status = ser.read().decode()
            if status == 'A':
                video.release()
                cv2.destroyAllWindows()
                break
    #ser.close()            # close port





#while(1):
#   ser.write("Data_Sending ".encode())     # write a string
#   time.sleep(1)   # 1 SEC delay


