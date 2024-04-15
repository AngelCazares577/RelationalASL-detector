# I will be testing the difference between a Convolutional Neural Network and LSTM(A type of Recurrent Neural Network)
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


from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout


sequenceLength = 25

input_shape = (3, 3, 1662, 32)

signs = np. array(['hello', "I", "Angel","A","J","Z"])

model = Sequential()

#TO-DO 

#want to use Time Distributed CNN layers 

#create a training generator that yields batches of sequences
# with shape (batch_size, sequenceLength, 300, 300, 3)
# and a corresponding generator for validation data.


