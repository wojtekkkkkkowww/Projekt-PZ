import os
import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_signs = 12
number_of_frames = 300

cap = cv2.VideoCapture(0)
for j in range(number_of_signs):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Zbieranie danych do znaku {}'.format(j+1))

    done = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Nacisnij f zeby zaczac', (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('f'):
            break

    result = []
    c = 0
    while c < number_of_frames:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_world_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                matrix = np.array([[landmark.x,landmark.y,landmark.z] for landmark in hand_landmarks.landmark],dtype=float)
                result.append(matrix)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(c)), frame)
        c += 1
    
    np.save(str(j)+".npy",np.array(result))


cap.release()
cv2.destroyAllWindows()