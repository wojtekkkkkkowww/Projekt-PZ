import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import glob
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

data_path = sys.argv[1]  
labels = os.listdir(data_path)

label_to_int = {label: i for i, label in enumerate(sorted(labels))}

selected_images = []

for label in labels:
    if label == 'nothing':
        continue

    image_files = glob.glob(os.path.join(data_path, label, "*.jpg"))
    selected_images.extend(random.sample(image_files, 3))

for image_file in selected_images:
    image = cv2.imread(image_file)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

    os.makedirs("landmarked_images", exist_ok=True)
    cv2.imwrite(f"landmarked_images/landmarked_{os.path.basename(image_file)}", image)