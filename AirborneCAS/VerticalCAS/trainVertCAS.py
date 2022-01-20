import numpy as np
#import theano
import math
import h5py
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

sys.path.append('../../')
import deepbayesHF

import tensorflow as tf
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

######## OPTIONS #########
ver = 4            # Neural network version
hu = 45            # Number of hidden units in each hidden layer in network
saveEvery = 10     # Epoch frequency of saving
totalEpochs = 200  # Total number of training epochs
trainingDataFiles = "TrainingData/VertCAS_TrainingData_v2_%02d.h5" # File format for training data
##########################

# The previous RA should be given as a command line input
if len(sys.argv) > 1:
    pra = int(sys.argv[1])
    print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
    f       = h5py.File(trainingDataFiles % pra,'r')
    X_train = np.array(f['X']).astype('float32')
    y_train       = np.array(f['y']).astype('float32')
    Q       = np.array(f['y'])
    means = np.array(f['means'])
    ranges=np.array(f['ranges'])
    min_inputs = np.array(f['min_inputs'])
    max_inputs = np.array(f['max_inputs'])
                       
    N,numOut = Q.shape
    print("Setting up Model")
    
    # Asymmetric loss function
    lossFactor = 40.0
    def asymMSE(y_true, y_pred):
        d = y_true-y_pred
        maxes = tf.argmax(y_true,axis=1)
        maxes_onehot = tf.one_hot(maxes,numOut)
        others_onehot = maxes_onehot-1
        d_opt = d*maxes_onehot 
        d_sub = d*others_onehot
        a = lossFactor*(numOut-1)*(tf.square(d_opt)+tf.abs(d_opt))
        b = tf.square(d_opt)
        c = lossFactor*(tf.square(d_sub)+tf.abs(d_sub))
        d = tf.square(d_sub)
        loss = tf.where(d_sub>0,c,d) + tf.where(d_opt>0,a,b)
        return tf.reduce_mean(loss)

    N,numInputs = X_train.shape
    N,numOut = Q.shape
    X_train = tf.cast(X_train, dtype=tf.float32)
    y_train = tf.argmax(y_train,axis=1)
    print("Setting up Model")

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    #loss = tf.keras.losses.CategoricalCrossentropy()
    # Use Keras to define our model
    WIDTH = 125
    model = Sequential()
    model.add(Dense(WIDTH, activation="relu", input_shape=(1, numInputs)))
    model.add(Dense(numOut, activation="softmax"))

    # Use deepbayesHF to define our optimizer
    optimizer = optimizers.VOGN()
    bayes_model = optimizer.compile(model, loss_fn=loss, learning_rate = 0.05,
                          epochs=25, batch_size=1024, decay=0.1,
                          # input_noise=0.05,
                          robust_train = 5, epsilon=0.05, rob_lam=0.0, classes=9,
                          inflate_prior=0.05, mode='classification') # select optimizer and set learning rate

    # Train and save BNN using deepbayesHF
    bayes_model.train(X_train, y_train, X_train, y_train)
    bayes_model.save('Posteriors/ROB_VCAS_BNN_%s'%(pra))
