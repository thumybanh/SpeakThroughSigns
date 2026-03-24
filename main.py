import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import uuid
import os as os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Changing frame Color
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


image = cv2.flip(image, 1)
image.flags.writeable = False
results = hands.process(image)

image.flags.writeable = True
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

if results.multi_hand_landmarks:
    for num, hand in enumerate(results.multi_hand_landmarks):
        # to change the colors you use this mp.drawing
        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=4), )

os.mkdir('Output images')