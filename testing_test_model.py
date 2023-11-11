import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

model = tf.keras.models.load_model('test.keras')

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


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

    if results.multi_hand_world_landmarks:
        for hand_landmarks in results.multi_hand_world_landmarks:

            matrix = np.array([[[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]], dtype=float)
            predictions = probability_model.predict(matrix)
            print(predictions)

    frame = cv2.flip(frame, 1)
    cv2.imshow('testowanie', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
