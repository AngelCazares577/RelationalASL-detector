import os 
import cv2 as cv
import mediapipe as mp 
import numpy as np 
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score



dataPath = os.path.join("extraction.data")

signs = np. array(['hello', "I", "Angel","A","J","Z"])

#number of frames used to detect action 
sequenceNum = 25

# "video" lengths 
sequenceLength = 25


signDict = {sign:num for num, sign in enumerate(signs)}

sequences, labels = [],[]
for sign in signs:
    for sequence in range(sequenceNum):
        window = []
        for frameNum in range(sequenceLength):
            res = np.load(os.path.join(dataPath, sign, str(sequence), "{}.npy".format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(signDict[sign])

#print(np.array(sequences).shape)
#print(np.array(labels).shape)


def categoryIs(y, num_classes=None):
    """Convert a class vector (integers) to binary class matrix."""
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes, dtype='float')[y]



x = np.array(sequences)

y = categoryIs(labels).astype(int) 
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences = True, activation = 'relu', input_shape = (25,1662)))
model.add(LSTM(128, return_sequences = True, activation = 'relu'))
model.add(LSTM(64, return_sequences = False, activation = 'relu' ))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(signs.shape[0], activation = 'softmax')) #FoftMax returns an array with a probability summed to one
#with the most density on the predicted sign

model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.fit(x_train,y_train, epochs = 700, callbacks = [tb_callback])

model.save('meshModel.h5')

model.load_weights('meshModel.h5')

yHat = model.predict(x_test)
yTrue = np.argmax(y_test, axis = 1).tolist()
yHat = np.argmax(yHat, axis = 1).tolist()

print(yTrue)
print(multilabel_confusion_matrix(yTrue,yHat))
print(accuracy_score(yTrue,yHat))


