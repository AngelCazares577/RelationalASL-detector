#importing dependencies 
import os 
import cv2 as cv
import mediapipe as mp 
import numpy as np 
import time
from matplotlib import pyplot as plt

#load model 
import tensorflow as tf
from tensorflow import keras 
from keras.models import load_model

model = load_model('meshModel.h5')


#mediapipe holsitic models 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def MP_detection(image,model):
     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
     image.flags.writeable = False
     results = model.process(image)
     image.flags.writeable = True
     image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
     return image, results

def draw_landmarks(image, results):

    mp_drawing.draw_landmarks(image,results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1))  # Custom style for connections)
    
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1))  # Custom style for connections))
    
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1))  # Custom style for connections))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1))  # Custom style for connections))

def data_extraction(results):
    
    faceMesh = np.zeros(468*3)
    poseMesh = np.zeros(33*4)
    leftMesh = np.zeros(21*3)
    rightMesh = np.zeros(21*3)
    if results.face_landmarks:
        faceMesh = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()

    if results.pose_landmarks:
        poseMesh = np.array([[r.x, r.y, r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten()

        # Check if left hand landmarks are detected and then process them
    if results.left_hand_landmarks:
        leftMesh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()

        # Check if right hand landmarks are detected and then process them
    if results.right_hand_landmarks:
        rightMesh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
    return np.concatenate([poseMesh, faceMesh, leftMesh, rightMesh])
        

sequence = []

signs = np. array(['hello', "I", "Angel","A","J","Z"])

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        frame = cv.flip(frame,1)
        
        #Make detections 
        image, results = MP_detection(frame, holistic)

        draw_landmarks(image,results)
        #prediction
        fullMesh = data_extraction(results)
        sequence.insert(0,fullMesh)
        sequence = sequence[:30]

        if len(sequence) == 30:
            prediction = model.predict(np.expand_dims(sequence, axis = 0))[0]
            print(signs[np.argmax(prediction)])

        cv.imshow( "Cv Stream ",image)
        if cv.waitKey(10) & 0xFF == ord('l'):
            break
    cap.release
    cv.destroyAllWindows()
    print(fullMesh.shape)
