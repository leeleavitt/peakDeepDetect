# In This experiment we are going to attempt to extract some Features to see if this improves performance
# On the last experiment our dataset was of the for
#[A, B, C]
# A: Number of traces/samples ~10,000
# B: Number of TimeSteps 120
# C: Number of features 1

# For this experiment we want to reduce the number of timesteps while increasin the number of Features
# A: Numver of traces/samples ~10,000
# B: Number of timeSteps 10
# C: Number of Features 3

# The features i would like to collect this time around are
# 1: Mean over 12 points
# 2: Standard Error over 12 points
# 3: Derivative over 12 Points
# Function to create the above described features

# Input to this function is
# 1: traces to calculate features
# 2: Size of Steps between the times
def featureMaker(traces, steps):
    #steps = 10
    # First initialize the empty data frame to fill up with the new 
    samples = traces.shape[0]
    timesteps = int(traces.shape[1] / steps)
    features = 4
    featureFrame = np.empty([samples, timesteps, features])


    rangeToCalc = np.arange(0, traces.shape[1]+1, steps)

    for i in range(len(rangeToCalc)-1):
        meanFeat = traces[:,rangeToCalc[i]:rangeToCalc[i+1]].mean(axis=1)
        stdFeat = traces[:,rangeToCalc[i]:rangeToCalc[i+1]].std(axis=1)
        semFeat = stats.sem(traces[:,rangeToCalc[i]:rangeToCalc[i+1]], axis=1)
        derivFeat = np.mean(np.gradient(traces[:,rangeToCalc[i]:rangeToCalc[i+1]], axis=1), axis=1)

        featureFrame[:, i, 0] = meanFeat
        featureFrame[:, i, 1] = stdFeat
        featureFrame[:, i, 2] = semFeat
        featureFrame[:, i, 3] = derivFeat
    
    return featureFrame

# Function to plot the Loss and Histories
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  
  #Loss Plotter
  fig, axs = plt.subplots(2)
  fig.suptitle(title)
  axs[0].plot(epochs, loss, 'b', label='Training loss')
  axs[0].plot(epochs, val_loss, 'r', label='Validation loss')
  axs[0].set_title('Loss')
  axs[0].legend()

  trainAcc = history.history['acc']
  valAcc = history.history['val_acc']

  axs[1].plot(epochs, trainAcc, 'b', label='Training Accuracy')
  axs[1].plot(epochs, valAcc, 'r', label='Validation Accuracy')
  axs[1].set_title('Accuracy')
  axs[1].legend()

  fig.show()

  fig.savefig(title+'.png', bbox_inches='tight')

# Load our Libraries
import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.style as stl 
from scipy import stats
stl.use('seaborn')
from sklearn.model_selection import train_test_split
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#This is where we are peakDeepDetect
superDir = os.getcwd()

#This is where we will be doing our work.
os.chdir('./data_in_csv')
main_dir  = os.getcwd() #make it my main dir
#Find only Directories
expDirs  = next(os.walk('.'))[1]

#Loop Through Each Application!
#for i in range(len(expDirs)):
for i in range(len(expDirs)):
    print("YIPPY KAY YAY!")
    print("Entering " + expDirs[i] +" Directory \n\n")
    os.chdir(expDirs[i])

    ######################################################################
    #First import the Traces
    traces = pd.read_csv("traces.csv", index_col=0)
    #some examples have na values get rid of
    traces = traces.dropna()
    tracesIndex = traces.index
    #Randomize!!
    tracesIndex = tracesIndex[np.random.permutation(len(tracesIndex))]
    traces = traces.loc[tracesIndex,]
    #This need to be a 3 dimensional numpy array
    traces = np.asarray(traces)
    
    tracesFeature = featureMaker(traces,10)

    # #Add the new Dimension
    # tracesFeature = tracesFeature[...,np.newaxis]

    #Load the Labels
    labels = pd.read_csv("labels.csv", index_col=0)
    #Load lables that match the traces above
    labels = labels.loc[tracesIndex,]
    #Convert to Category
    labels = labels.iloc[:,0].astype('category')
    #convert to np array
    labels = np.asarray(labels)


    #Create Train and Validation Set
    val = int(np.ceil(tracesFeature.shape[0]*.33))
    trainSize = tracesFeature.shape[0] - val 

    x_train  = tracesFeature[:trainSize,...]
    y_train = labels[:trainSize]

    x_test = tracesFeature[trainSize:,...]
    y_test = labels[trainSize:]

    # Now DO what we need for the import to LSTM
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test = test.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    model = Sequential([
        tf.keras.layers.LSTM(12, input_shape = tracesFeature.shape[-2:]),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss="binary_crossentropy",
                metrics=['acc'])

    model.summary()

    EVALUATION_INTERVAL = 500
    EPOCHS = 50

    history = model.fit(train, epochs = EPOCHS, 
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data = test,
                        validation_steps=50)

    # history = model.fit(x_train, y_train, epochs=25, 
    #                     validation_data=(x_test,y_test ))

    print("This is the loss vs Accuracy for" + expDirs[i])
    plot_train_history(history, expDirs[i]+"_lstm.experiment2")     
    
    #Save the ModelYYYYYYYYYYY
    model.save(expDirs[i]+'exp2.h5')

    ##########################################################
    #Now that we have a model that works fairly well 
    #lets do some data analysis

    #These are Predicted Values
    labsPred = model.predict_classes(tracesFeature)
    labsPred = pd.DataFrame(labsPred) #convert to df

    #convert real to DataFrame
    labs = pd.DataFrame(labels)

    realTest = pd.concat([labs, labsPred], axis=1)

    realTest.columns = ['Real', "Predicted"]

    realTest = realTest.set_index(tracesIndex)

    realTest.to_csv('lstm.experiment2_labcomp.csv')

    #These are Predicted Values
    realVsPredCT = pd.crosstab(np.asarray(labsPred).flatten(), np.asarray(labs).flatten(), rownames=['pred'], colnames=['real'])
    
    print(realVsPredCT)

    del model

    os.chdir(main_dir)

