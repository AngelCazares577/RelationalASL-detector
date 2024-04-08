#importing dependencies 
import os 
import cv2 as cv
import mediapipe as mp 
import numpy as np 
import time
from matplotlib import pyplot as plt


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
    #setting default values to zero and they will only change if all relevant points are detected as to avoud 
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




dataPath = os.path.join("extraction.data")

signs = np. array(['hello', "I", "Angel"])

#number of frames used to detect action 
sequenceNum = 30

# "video" lengths 
sequenceLength = 30

for sign in signs:
    for sequence in range(sequenceNum):
        try:
            os.makedirs(os.path.join(dataPath, sign, str(sequence)))
        except:
            pass



stream = cv.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sign in signs:
        for sequence in range(sequenceNum):
            for frameNum in range(sequenceLength):
                ret,frame = stream.read()
                frame = cv.flip(frame,1)
                
                #Make detections 
                image, results = MP_detection(frame, holistic)
                print(results.pose_landmarks)

                draw_landmarks(image,results)

                #Keeping track of video number and giving a wait time between videos
                if frameNum == 0: 
                    cv.putText(image, 'DataBase Collection Starting...', (120,200), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(image, 'SIGN: {} Video Number {}'.format(sign, sequence), (15,12), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    # Show to screen
                    cv.imshow('CV Stream', image)
                    cv.waitKey(1500)
                else: 
                    cv.putText(image, 'SIGN: {} Video Number {}'.format(sign, sequence), (15,12), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    # Show to screen
                    cv.imshow('Stream', image) 
                
                #call function to extract all keypoints and coordinate data
                keypoints = data_extraction(results)
                pyPath= os.path.join(dataPath, sign, str(sequence), str(frameNum))
                np.save(pyPath, keypoints)

                if cv.waitKey(10) & 0xFF == ord('l'):
                    break
                if cv.waitKey(10) & 0xFF == ord('L'):
                    break               

    stream.release
    cv.destroyAllWindows()

