import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import time
import sys
model = tf.keras.models.load_model(f"models/perm_model1.keras")

SUPER = False


key = np.load("key.npy")


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

language = [i + 1 for i in range(13)]

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

            input = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
            input = np.array(input)
           
            input = input.flatten() 
            input = input[key] # nakladanie permutacji
            input = np.array([input])
           
            
            
            predictions = probability_model.predict(input)
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