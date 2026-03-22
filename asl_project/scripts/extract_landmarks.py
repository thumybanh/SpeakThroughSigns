import mediapipe as mp
import cv2
import os
import csv


# Mediapipe 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 1)


# CV to read the hand images in data. 
dataset_path = '../data/asl_alphabet/asl_alphabet_train'
folders = os.listdir(dataset_path)


allData = []
for folder in folders: 
    if folder == '.DS_Store':
        continue
    alphabet_folders = os.listdir(os.path.join(dataset_path, folder)) #locate each alphabet folder
    for image in alphabet_folders: 
        row = [folder]
        img = cv2.imread(os.path.join(dataset_path, folder, image))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # use mediapipe to detect hand landmarks
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark: 
                row.extend([lm.x, lm.y, lm.z])
            allData.append(row)

csv_file_path = '../data/landmarks.csv'          
with open(csv_file_path, 'w') as f: 
    writer = csv.writer(f)
    writer.writerows(allData)



