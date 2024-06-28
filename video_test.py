import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import time
import argparse
from collections import deque
from models import GestureRecognizerModel,sequentials

parser = argparse.ArgumentParser(prog='Video test for models')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-s', '--sequential', action='store_true')

args = parser.parse_args()


if(args.model is None):
    print('see -h')
    exit()

LITE = False
if(args.model.endswith('tflite')):
    LITE = True

model = GestureRecognizerModel(None,args.model.split(".")[0],args.permuted,args.sequential,LITE)

SUPER = False
if(args.model.startswith('supermodel')):
    SUPER = True
    submodels = [GestureRecognizerModel(None,f'model{i}',args.permuted,sequentials[i-1],args.lite) for i in range(1,4)]


language = [i + 1 for i in range(28)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

    predicted_sign_index = -1
    sureness = 0
    if results.multi_hand_world_landmarks:
        for hand_landmarks in results.multi_hand_world_landmarks:

            startTime = time.time()

            matrix = np.array([[[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]], dtype=float)

            if(SUPER):
                res = []
                for subm in submodels:
                    res.append( subm.predict(matrix) )
                matrix = np.array(np.concatenate(res))

            predictions = model.predict(matrix)

            predicted_sign_index = np.argmax(predictions[0])
            sureness = predictions[0][predicted_sign_index]
            print("Time: ",time .time() - startTime)
            #if sureness < 0.9:
            #    predicted_sign_index = 12
            #    sureness = 0

    frame = cv2.flip(frame, 1)

    cv2.putText(frame, str(language[predicted_sign_index]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(sureness), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('testowanie', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()