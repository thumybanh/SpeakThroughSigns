import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('../data/landmarks.csv', header=None)
df = df[~df[0].isin(['del', 'nothing', 'space'])]
# alphabet letter 
y = df[0]
# numbers after the letter
x = df.iloc[: , 1:] # df[all rows, give me columns starting from index 1]

le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# building the neural network using keras: relu = helps networks find patterns through hidden layers ; soft-max = output layer where it coonverst 26 neuron values into probability that adds up to 100%
model = tf.keras.Sequential([
    # LAYER 1 
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)), #128 neurons, make the neural network recognize the pattern instead of memorizing the exact points
    # LAYER 2 
    tf.keras.layers.Dropout(0.3), #fixes overfitting problem by turning off some of neurons during training 0.3 means 30% of the neurons randomly switch off each session
    #LAYER 3 
    tf.keras.layers.Dense(64, activation='relu'),
    #LAYER 4
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'] )
model.fit(x_train,y_train,epochs=30,validation_data=(x_test, y_test))

model.save('../model/asl_model.keras')

with open('../model/label_encoder.pkl', 'wb') as f: 
    pickle.dump(le, f)



