import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            landmarks_to_draw = [4, 8]
            cords = []
            for idx in landmarks_to_draw:
                landmark_point = hand_landmarks.landmark[idx]
                cx, cy = int(landmark_point.x * frame.shape[1]), int(landmark_point.y * frame.shape[0])
                cords.append((cx,cy))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            cv2.line(frame,cords[0],cords[1],(0,0,255),1)

            length = math.sqrt((cords[0][0] - cords[1][0])**2 + (cords[0][1] - cords[1][1])**2)
            val = np.interp(length,[20,200],[0,100])
            print(val)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
