#importing dependencies 
import os 
import cv2 
import mediapipe as mp 
import numpy as np 
import time
from matplotlib import pyplot as plt


#mediapipe holsitic models 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def MP_detection(image,model):
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     image.flags.writeable = False
     results = model.process(image)
     image.flags.writeable = True
     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
     return image, results

def draw_landmarks(image, results):
    
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1))  # Custom style for connections))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1))  # Custom style for connections))

def data_extraction(results):
    #setting default values to zero and they will only change if all relevant points are detected as to avoud 
    leftMesh = np.zeros(21*3)
    rightMesh = np.zeros(21*3)

        # Check if left hand landmarks are detected and then process them
    if results.left_hand_landmarks:
        leftMesh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()

        # Check if right hand landmarks are detected and then process them
    if results.right_hand_landmarks:
        rightMesh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
    return np.concatenate([leftMesh, rightMesh])


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classSize = 3
setSize = 150

cap = cv2.VideoCapture(0)
for i in range(classSize):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))

    print('Collecting data for class {}'.format(i))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'press L when ready:)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('L'):
            break

    counter = 0
    while counter < setSize:
        MP_detection(frame)
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()