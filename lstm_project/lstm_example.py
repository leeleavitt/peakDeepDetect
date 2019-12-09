import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.style as stl 
import os
stl.use('seaborn')

from sklearn.model_selection import train_test_split


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


df = pd.read_csv(csv_path)
