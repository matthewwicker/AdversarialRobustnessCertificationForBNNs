import argparse
import os
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# Stuff we made
import deepbayesHF
from deepbayesHF import optimizers
from datasets import Dataset
import sklearn

parser = argparse.ArgumentParser()
parser.add_argument(
        "--dataset",
        help="Pick a dataset")
args = parser.parse_args()
dataset_string = args.dataset

if(dataset_string == "kin8nm1"):
    lr = 4.0; bs = 256; prior=0.01
    epochs = 1500

elif(dataset_string == "concrete1"):
    lr = 1.5; bs = 256; prior=1.0
    epochs = 2000

elif(dataset_string == "boston1"):
    lr = 2.25; bs = 256; prior=0.25
    epochs = 3000

elif(dataset_string == "wine1"):
    lr = 7.5; bs = 256; prior=0.05
    epochs = 1500

elif(dataset_string == "powerplant1"):
    lr = 15.0; bs = 256; prior=0.01
    epochs = 1250


elif(dataset_string == "naval1"):
    lr = 5.0; bs = 256; prior=0.01
    epochs = 1500


elif(dataset_string == "energy1"):
    lr = 1.25; bs = 256; prior=1.0
    epochs = 2000


elif(dataset_string == "yacht1"):
    lr = 4.5; bs = 64; prior=0.01
    epochs = 2750
else:
    print("Dataset not found... exiting")
    sys.exit(0)

data = Dataset(dataset_string)
from sklearn.preprocessing import StandardScaler

X_train = data.train_set.train_data
y_train = data.train_set.train_labels.reshape(-1,1)

X_test = data.test_set.test_data
y_test = data.test_set.test_labels.reshape(-1,1)

X_scaler = StandardScaler()
X_scaler.fit(X_train)

y_scaler = StandardScaler()
y_scaler.fit(y_train)

X_train, X_test = X_scaler.transform(X_train), X_scaler.transform(X_test)
y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)

# Dataset information:
in_dims = X_train.shape[1]
print(X_train.shape)
print(y_train.shape)

# This is also known as the loss, dont get worried about the Bayesian terminology of things
likelihood = tf.keras.losses.MeanSquaredError()
opt = optimizers.VariationalOnlineGuassNewton()

# A small architecture means fast verification :-)
model = Sequential()
model.add(Dense(10, activation="tanh", input_shape=(1, in_dims)))
model.add(Dense(1, activation="linear"))

if not os.path.exists('TrainingLogs'):
    os.makedirs('TrainingLogs')

bayes_model = opt.compile(model, loss_fn=likelihood, 
                          epochs=epochs, learning_rate=lr, batch_size=bs,
                          inflate_prior=prior, mode='regression',
                          log_file='TrainingLogs/%s_training.log'%(dataset_string))

bayes_model.train(X_train, y_train, X_test, y_test)

bayes_model.save("PosteriorScaled/VOGN_%s_Model"%(dataset_string))
