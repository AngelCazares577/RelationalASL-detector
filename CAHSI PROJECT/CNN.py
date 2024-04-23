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
from keras.layers import Input


from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout
from scipy.ndimage import gaussian_filter

#ok so like CNN does not work on data that is not grid-like. Sooo.. gridMap it is 
def keypoints_to_heatmap(keypoints, image_size=(300, 300), sigma=10):
    """
    Convert keypoints to a spatial heatmap.
    keypoints - Array of shape (num_keypoints, 3) for (x, y, visibility)
    image_size - Size of the output heatmap (height, width)
    sigma - Standard deviation for Gaussian
    """
    heatmaps = np.zeros((image_size[0], image_size[1], keypoints.shape[0]), dtype=np.float32)
    for i, (x, y, visibility) in enumerate(keypoints):
        if visibility > 0: 
            x = int(x * image_size[1])
            y = int(y * image_size[0])
            if x >= 0 and y >= 0 and x < image_size[1] and y < image_size[0]:
                # Set a point at the keypoint location
                heatmaps[y, x, i] = 1
    # Apply Gaussian blur to each channel
    for i in range(keypoints.shape[0]):
        heatmaps[:, :, i] = gaussian_filter(heatmaps[:, :, i], sigma=sigma)

    return heatmaps

