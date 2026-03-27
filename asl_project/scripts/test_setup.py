import mediapipe as mp
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import sklearn
import os


print("mediapipe:", mp.__version__)
print("tensorflow:", tf.__version__)
print("opencv:", cv2.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)


dataset_path = '../data/asl_alphabet/asl_alphabet_train'
if os.path.exists(dataset_path):
    folders = os.listdir(dataset_path)
    print(f'Dataset found: {len(folders)} letters folders detected')
    print('letters found:', sorted(folders))
else: 
    print("\nDataset NOT found. Check your data/ folder.")

