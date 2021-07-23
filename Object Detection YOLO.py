import cv2
import numpy as np

# Load Yolo using Deep neural network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Loading Classes Names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Getting the name of three O/P layers of network along with their Indexes
output_layers = net.getUnconnectedOutLayersNames()

# Initializing the camera
video = cv2.VideoCapture(0)
while True:
    # Loading image
    ret, frame = video.read()
    if not ret:
        print("Failed to Initialize Camera")
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        break

    img = frame
    #height, width, channels = img.shape

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

    #Printing Detected Object Names
    for i in range(len(class_ids)):
        print(str(classes[class_ids[i]])+" detected")

video.release()
cv2.destroyAllWindows()