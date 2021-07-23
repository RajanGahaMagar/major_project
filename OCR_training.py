import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

import pickle

###########################################
path = "DataSets_(0-9)Fnt"
testRatio = 0.2
valRatio = 0.2
imageDiamensions = (32, 32, 3)
batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 2000

###########################################

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Datasets Found: ", myList)
print("Total No Of Classes Detected: ", len(myList))
noOfClasses = len(myList)
print("Importing Classes... ")

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (imageDiamensions[0], imageDiamensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
    count += 1
print(" ")

images = np.array(images)
classNo = np.array(classNo)
print("Total Images in Images List, h_size, v_size, Color_Channels: ", images.shape)
print("Total IDs in classNo List: ", classNo.shape)

####### Splitting the Data #######
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=valRatio)
print("Training List: ", x_train.shape)
print("Test List: ", x_test.shape)
print("Validation List: ", x_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
print("Samples Of each Class: ", numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number Of Images")
plt.show()


######## Preprocessing of images #########
def preProcessing(img):
    # img = cv2.equalizeHist(img)         # makes the lighting of the image distribute evenly
    img = cv2.medianBlur(img, 5)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = img / 255                     # normalization of gray image (0-255)=>(0-1)
    return img


x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))
x_validation = np.array(list(map(preProcessing, x_validation)))

######## Adding Depth 1 in images #########
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

####### Augmentation of Images ############
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

####### One hot encoding #########  =>used to categorize data based on labels & convert them into specific binary representation
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


######## Defining a Model using Concept of LinNet Model ##########
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDiamensions[0],imageDiamensions[1],1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))     #this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs.
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))     # helps to reduce overfading
    model.add(Flatten())        #Flattening is the conversion of data into 1D array for inputting it into next layer
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

############ Start Training ############
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batchSizeVal),
                              steps_per_epoch=stepsPerEpochVal,
                              epochs=epochsVal,
                              validation_data=(x_validation, y_validation),
                              shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

pickle_out = open("trained_(0-9)Fnt.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
