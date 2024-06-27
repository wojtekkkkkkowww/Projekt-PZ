import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import glob

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

data_path = sys.argv[1]  
labels = os.listdir(data_path)

label_to_int = {label: i for i, label in enumerate(sorted(labels))}

for label in labels:
    image_files = glob.glob(os.path.join(data_path, label, "*.jpg"))

    current_data = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            print(f"Processing {image_file}")
            for hand_landmarks in results.multi_hand_landmarks:
                matrix = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark], dtype=float)
                current_data.append(matrix)
        else:
            print(f"No hand landmarks found in {image_file}")

    if current_data:
        np_data = np.array(current_data)
        np.random.shuffle(np_data)

        test_data, train_data = np_data[:len(np_data)//5], np_data[len(np_data)//5:]
        train_data_splits = np.array_split(train_data, 3)

        os.makedirs(f"data/ASLtest", exist_ok=True)
        np.save(f"data/ASLtest/{label_to_int[label]}.npy", test_data)
        for i, train_data_split in enumerate(train_data_splits):
            os.makedirs(f"data/model{i+1}/ASLtrain", exist_ok=True)
            np.save(f"data/model{i+1}/ASLtrain/{label_to_int[label]}.npy", train_data_split)