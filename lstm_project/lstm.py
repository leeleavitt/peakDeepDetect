import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.style as stl 
stl.use('seaborn')
from sklearn.model_selection import train_test_split

######################################################################
#First import the K.40mM 
traces = pd.read_csv("./data_in_csv/K.40mM/traces.csv", index_col=0)
#some examples have na values get rid of
traces = traces.dropna()
tracesIndex = traces.index
#Randomize!!
tracesIndex = tracesIndex[np.random.permutation(len(tracesIndex))]
traces = traces.loc[tracesIndex,]
#This need to be a 3 dimensional numpy array
traces = np.asarray(traces)
#Add the new Dimension
traces = traces[...,np.newaxis]

#Load the Labels
labels = pd.read_csv("./data_in_csv/K.40mM/labels.csv", index_col=0)
#Load lables that match the traces above
labels = labels.loc[tracesIndex,]
#Convert to Category
labels = labels.iloc[:,0].astype('category')
#convert to np array
labels = np.asarray(labels)

#Create Train and Validation Set
val = int(np.ceil(traces.shape[0]*.33))
trainSize = traces.shape[0] - val 

x_train  = traces[:trainSize,...]
y_train = labels[:trainSize]

x_test = traces[trainSize:,...]
y_test = labels[trainSize:]

# Now DO what we need 
BATCH_SIZE = 256
BUFFER_SIZE = 10000
train = tf.data.Dataset.from_tensor_slices((traces, labels))
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test = test.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



#This Doesn't work for 3 dimension datasets
#X_train, X_test, y_train, y_test = train_test_split(traces, labels, test_size=0.33)

#Using this as a guide
#https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

# Function to transform data set into 10 peices of 
# Mean
# Standard Deviation


#To feed into LSTM we need 3 Dimensions
# 1 # of samples (11063)
# 2 # of features (2 mean and Standar Deviation)
# 3 # of timesteps

#This Helps to guide the model and loss
#https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model = Sequential([
    tf.keras.layers.LSTM(100, input_shape = traces.shape[-2:]),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['acc'])

model.summary()

EVALUATION_INTERVAL = 200
EPOCHS = 10

history = model.fit(train, epochs = EPOCHS, 
                    steps_per_epoch=EVALUATION_INTERVAL,
                    validation_data = test,
                    validation_steps=50)

history = model.fit(x_train, y_train, epochs=25, 
                    validation_data=(x_test,y_test ))

                    