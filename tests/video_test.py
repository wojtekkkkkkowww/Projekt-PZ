import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import time
import argparse
from collections import deque

parser = argparse.ArgumentParser(prog='Video test for models')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-s', '--sequential', action='store_true')

args = parser.parse_args()

if(args.model is None):
    print('see -h')
    exit()

MODEL_LITE = False
if(args.model.endswith('tflite')):
    interpreter = tf.lite.Interpreter(model_path=f'models/{args.model}')
    signature = interpreter.get_signature_runner()
    MODEL_LITE = True
else:
    model = tf.keras.saving.load_model(f"models/{args.model}")

SUPER = False
if(args.model.startswith('supermodel')):
    if(args.lite):
        interpreters = [tf.lite.Interpreter(model_path=f'models/model{i}.tflite') for i in range(1,4)]
        signatures = [i.get_signature_runner() for i in interpreters]
    else:
        models = [tf.keras.saving.load_model(f"models/model{i}.keras") for i in range(1,4)]
    SUPER = True


language = [i + 1 for i in range(28)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SEQ_LEN = 10
cap = cv2.VideoCapture(0)
seq_data = deque([np.zeros((21*3)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

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
                for m in range(3):
                    if(args.permuted):
                        key = np.load(f'keys/model{m+1}_key.npy')
                        matrix = np.array(matrix.flatten()[key]).reshape(21,3)

                    if(args.lite):
                        atr = list(signatures[m].get_input_details().keys())[0]
                        p = signatures[m](**{atr:np.array([matrix], dtype=np.float32)})
                        p = p[list(p.keys())[0]]
                    else:
                        p = models[m].predict(matrix)

                    res.append(p)
                matrix = np.array(np.concatenate(res))
            elif(args.permuted):
                key = np.load(f'keys/{args.model.split(".")[0]}_key.npy')
                matrix = np.array(matrix.flatten()[key]).reshape(21,3)

            if(args.sequential):
                seq_data.appendleft(matrix.flatten())
                matrix = np.array(seq_data)
            
            if(MODEL_LITE):
                atr = list(signature.get_input_details().keys())[0]
                predictions = signature(**{atr:np.array([matrix], dtype=np.float32)})
                predictions = predictions[list(predictions.keys())[0]]
            else:
                predictions = model.predict(np.array(matrix))


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