import numpy as np
#import theano
import math
import h5py
import sys

sys.path.append('../../')
import deepbayesHF

import tensorflow as tf
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
######## OPTIONS #########
ver = 4            # Neural network version
hu = 45            # Number of hidden units in each hidden layer in network
saveEvery = 10     # Epoch frequency of saving
totalEpochs = 200  # Total number of training epochs
trainingDataFiles = "./TrainingData/VertCAS_TrainingData_v2_%02d.h5" # File format for training data
nnetFiles = "./networks/VertCAS_pra%02d_v%d_45HU_%03d.nnet" # File format for .nnet files
##########################

def accAcc(y_true, y_pred):
    #maxesPred = tf.argmax(y_pred,axis=1)
    #inds = tf.argmax(y_true,axis=1)
    #diff = tf.cast(tf.abs(inds-maxesPred),dtype='float64')
    #ones = tf.ones_like(diff,dtype='float64')
    #zeros= tf.zeros_like(diff,dtype='float64')
    #l = tf.where(diff!=0,ones,zeros)
    #return tf.reduce_mean(l)
    maxpreds = tf.argmax(y_pred, axis=1)
    maxtrues = tf.argmax(y_true, axis=1)
    equality = tf.math.equal(maxpreds, maxtrues)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy

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
    
    bayes_model = PosteriorModel('Posteriors/NEW_VCAS_BNN_%s'%(pra))
    print("*****************************************")

    y_pred = bayes_model.predict(X_train[0:1000000])
    acc = accAcc(y_train[0:1000000], y_pred)
    print(acc, pra, ver)

