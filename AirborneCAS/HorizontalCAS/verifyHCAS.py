import numpy as np
import math
import tensorflow as tf
import h5py
import sys
import os

import tensorflow as tf
from tensorflow.keras.models import *

from tensorflow.keras.layers import *
sys.path.append('../../')
import deepbayesHF
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import prob_veri


os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

######## OPTIONS #########
ver = 6              # Neural network version
table_ver = 6        # Table Version
hu = 25              # Number of hidden units in each hidden layer in network
saveEvery = 1000     # Epoch frequency of saving
totalEpochs = 3000   # Total number of training epochs
trainingDataFiles = "./TrainingData/HCAS_rect_TrainingData_v%d_pra%d_tau%02d.h5" # File format for training data
nnetFiles  = "./networks/HCAS_rect_v%d_pra%d_tau%02d_%dHU.nnet" # File format for .nnet files
modelFiles = "./models/HCAS_rect_v%d_pra%d_tau%02d_%dHU.ckpt" # File format for .nnet files
tbFiles = "./tensorboard/HCAS_rect_v%d_pra%d_tau%02d_%dHU"
##########################

# Function to compute network accuracy given. However, this does not 
# add the online costs that were used to generate the correct minimum cost index, so
# these accuracies are only an estimate
def custAcc(y_true,y_pred):
    maxesPred = tf.argmax(y_pred,axis=1)
    inds = tf.argmax(y_true,axis=1)
    diff = tf.cast(tf.abs(inds-maxesPred),dtype='float64')
    ones = tf.ones_like(diff,dtype='float64')
    zeros= tf.zeros_like(diff,dtype='float64')
    l = tf.where(diff<0.5,ones,zeros)
    return tf.reduce_mean(l)

def accAcc(y_true, y_pred):
    maxesPred = tf.argmax(y_pred,axis=1)
    inds = tf.argmax(y_true,axis=1)
    diff = tf.cast(tf.abs(inds-maxesPred),dtype='float64')
    ones = tf.ones_like(diff,dtype='float64')
    zeros= tf.zeros_like(diff,dtype='float64')
    l = tf.where(diff!=0,ones,zeros)
    return tf.reduce_mean(l)

# The previous RA should be given as a command line input
if len(sys.argv) > 2:
    pra = int(sys.argv[1])
    tau = int(sys.argv[2])

    property = int(sys.argv[3])
    gpu = -1
    print("Verifying the following: ")
    print("Posterior Path: \t Posteriors/HCAS_BNN_%s_%s"%(pra, tau))
    print("Log path: \t \t ")
    print("Consistency property: %s"%(property))
    #if len(sys.argv)>3:
    #    gpu = int(sys.argv[3])

    print("Loading Data for HCAS, pra %02d, Network Version %d" % (pra, ver))
    f       = h5py.File(trainingDataFiles % (table_ver,pra,tau),'r')
    X_train = np.array(f['X']).astype('float32')
    y_train       = np.array(f['y']).astype('float32')
    Q       = np.array(f['y']).astype('float32')
    means = np.array(f['means'])
    ranges=np.array(f['ranges'])
    mins = np.array(f['min_inputs'])
    maxes = np.array(f['max_inputs'])

    min_inputs      = ",".join(np.char.mod('%f', mins))
    max_inputs     = ",".join(np.char.mod('%f', maxes))
    means      = ",".join(np.char.mod('%f', means))
    ranges     = ",".join(np.char.mod('%f', ranges))

    N,numInputs = X_train.shape
    N,numOut = Q.shape

    classes = tf.argmax(y_train,axis=1)
    inputs = np.argwhere(classes == property)


    print("Setting up Verification")
    print("Verifying %s of %s inputs"%(len(inputs), len(classes)))

    for i in range(len(inputs)):
        INDEX = inputs[i]

        bayes_model = PosteriorModel("Posteriors/ROB_HCAS_BNN_%s_%s"%(pra, tau))
        y_pred = bayes_model.predict(np.asarray([X_train[INDEX]]))
        if(np.argmax(y_pred) != property):
            print("misclassified")
            continue
        #skip = custAcc([y_train[INDEX]], y_pred)
        #if(skip == 0):
        #    print("initially misclassified")
        #    continue # prob 0

        TRUE_VALUE = property
        def predicate_safe(iml, imu, ol, ou):
            v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
            v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
            v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
            worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
            if(np.argmax(worst_case) == TRUE_VALUE):
                return True
            else:
                return False

        EPSILON = 0.025
        MAXDEPTH = 3
        SAMPLES = 3
        MARGIN = 2.75
        inp_upper = X_train[INDEX]+(EPSILON) #np.clip(np.asarray(X_train[INDEX]+(EPSILON)), -100000, 100000)
        inp_lower = X_train[INDEX]-(EPSILON) #np.clip(np.asarray(X_train[INDEX]-(EPSILON)), -100000, 100000)

        p_lower = prob_veri(bayes_model, inp_lower, inp_upper, MARGIN, SAMPLES, predicate=predicate_safe, depth=MAXDEPTH)
        print("Initial Safety Probability: ", p_lower)

        #Save result to log files
