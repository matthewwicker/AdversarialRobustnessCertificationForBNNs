import numpy as np
import theano
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
trainingDataFiles = "./TrainingData/VertCAS_TrainingData_v2_%02d.h5" # File format for training data
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


    # Define model architecture
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=4))
    model.add(Dense(numOut))

    # Use deepbayesHF to define our optimizer
    optimizer = optimizers.VOGN()
    bayes_model = optimizer.compile(model, loss_fn=asymMSE, learning_rate = 0.001,
                          epochs=75, batch_size=1024,
                          inflate_prior=0.1, mode='regression') # select optimizer and set learning rate
    #bayes_model.valid_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="extra_acc")
    #bayes_model.train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="extra_acc")

    # Train and save BNN using deepbayesHF
    bayes_model.train(X_train, y_train, X_train, y_train)
    bayes_model.save('Posteriors/VOGN_VCAS_BNN_%s'%(pra))
