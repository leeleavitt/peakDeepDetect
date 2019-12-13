# peakDeepDetect
This is a series of experiments to develop a neural network which will automatically detect artificially provoked responses from neurons. To understand this project in greater depth view the documents in the `Documents` folder.

We suggest you clone this repository
```
git clone https://github.com/leeleavitt/peakDeepDetect.git
```
Additionally when working with any code it is important to make the folder `peakDeepDetect` the top directory. We highly reccoment using the [anaconda distribution](https://www.anaconda.com/distribution/), and also install and use **VScode**. This will helps alleviate setting the working directories to run these experiments. If not follow these examples below `directory` is the location of this respository on your local machine. For example on my machine `C:/Users/leele/Documents/peakDeepDetect/` is where this directory is located. To set these directories at the top level, 

In `Python`
```
import os
os.chdir('directory/peakDeepDetect')
```
Some portions of this code, especially the visualization are done with `R`, it is also important to ensure that this folder is the directory focus. To change the directory in `R` use,
```
setwd('directory/peakDeepDetect')
```
## Folders and Descriptions
* `data_in_csv` : This folder contains all data for each response type. Within each folder there are a variety of different file types, this is mainly the dump location for the output of the experiments run. For example in the first folder `AITC.100uM` each experiment/attempt as a variety of different files. Each attempt has the prefix attached before.
    + `traces.csv` and `lables.csv`: contain the data to be trained on
    + `exp2.AITC.100uM_lstm.experiment1.png`: contain the loss and accuracy results from the first lstm experiment.
    + `exp2.AITC.100uMexp1.h5`: This is the model trained on the first experiments.
    + `exp2.lstmExp1OnesAsZeros`: These type of files are the visualizations for the misclassified cells.

* `Documents` This contains the formal documents describing these experiments.

* `Attempt 2` : The main file that is most important in this folder is the `lstm_experiment1.py`. This contains the experiment and the architecture. It is well documented, and easy to follow. Since it rotates through many directories, if it were to fail or stop, it is important to run it again in a fresh session. To run this ensure your are in the correct directory. Then simply run `python ./Attempt \4/lstm_experiment2.py`. This should execute. If this does not work, open python and simply copy and paste the code into the console.

* `Attempt 3`: This contains a jupyter notebook showing the execution of this experiment.

* `Attempt 4`: This is the final experiment. For more information on this project please see `Documents/Final_Report.pdf`





