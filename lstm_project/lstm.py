import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.style as stl 
stl.use('seaborn')

from sklearn.model_selection import train_test_split

#First import the K.40mM 
traces = pd.read_csv("./data_in_csv/K.40mM/traces.csv", index_col=0)
#some examples have na values get rid of
traces = traces.dropna()

#Load the Labels
labels = pd.read_csv("./data_in_csv/K.40mM/labels.csv", index_col=0)
labels = labels.loc[traces.index,]
labels = labels.iloc[:,0].astype('category')

X_train, X_test, y_train, y_test = train_test_split(traces, labels, test_size=0.33)

#Using thi as a guide
#https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

# Function to transform data set into 10 peices of 
# Mean
# Standard Deviation


#To feed into LSTM we need 3 Dimensions
# 1 # of samples (11063)
# 2 # of features (2 mean and Standar Deviation)
# 3 # of timesteps











from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model = Sequential()
model.add(LSTM(1,input_shape=(120,1)))






# model = Sequential([
#     tf.keras.layers.Flatten(input_shape = X_train.iloc[0].shape),
#     tf.keras.layers.LSTM(1),
# ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['Accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=25, 
                    validation_data=(X_test,y_test ))

                    