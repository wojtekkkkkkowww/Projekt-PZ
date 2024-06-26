import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sys
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

data_path = sys.argv[1]  
data = pd.read_csv(data_path)

output_dir = f"data/{'mint_train' if 'train' in data_path else 'minst_test'}"
os.makedirs(output_dir, exist_ok=True)

data_dict = {}

for i, row in data.iterrows():
    print(f'{i}')
    label = row['label'] 
    image = row[1:].values.reshape(28, 28).astype(np.uint8)

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            matrix = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark], dtype=float)
            
            if label in data_dict:
                data_dict[label].append(matrix)
            else:
                data_dict[label] = [matrix]

for label, data in data_dict.items():
    print(f"{len(data)}")
    np.save(f"{output_dir}/{label}.npy", np.array(data))