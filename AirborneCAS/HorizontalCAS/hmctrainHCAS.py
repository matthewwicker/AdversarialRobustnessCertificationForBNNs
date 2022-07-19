import numpy as np
import math
import tensorflow as tf
import h5py
import sys
import os

sys.path.append('../../')
import deepbayesHF

import tensorflow as tf
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

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

# Custom tensorflow session. Sets up training with either a cpu, gpu, or multiple gpus
def get_session(gpu_ind,gpu_mem_frac=0.45):
    """Create a session that dynamically allocates memory."""
    if gpu_ind[0]>-1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(np.char.mod('%d', gpu_ind))
        config = tf.ConfigProto(device_count = {'GPU': len(gpu_ind)})
        config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac
        session = tf.Session(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        session = tf.Session()
    return session

# Function to compute network accuracy given. However, this does not 
# add the online costs that were used to generate the correct minimum cost index, so
# these accuracies are only an estimate
# Note from MRW: This doesnt make sense as a metric...
def custAcc(y_true,y_pred):
    maxesPred = tf.argmax(y_pred,axis=1)
    inds = tf.argmax(y_true,axis=1)
    diff = tf.cast(tf.abs(inds-maxesPred),dtype='float64')
    ones = tf.ones_like(diff,dtype='float64')
    zeros= tf.zeros_like(diff,dtype='float64')
    l = tf.where(diff<0.5,ones,zeros)
    return tf.reduce_mean(l)


# The previous RA should be given as a command line input
if len(sys.argv) > 2:
    pra = 0 #int(sys.argv[1])
    tau = 0 #int(sys.argv[2])
    gpu = -1
    print("Posteriors/HCAS_BNN_%s_%s"%(pra, tau))
    if len(sys.argv)>3:
        gpu = int(sys.argv[3])

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
    X_train = tf.cast(X_train, dtype=tf.float64)
    y_train = tf.argmax(y_train,axis=1)

    print("Setting up Model")

    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    y_train = tf.one_hot(y_train, depth=numOut)
    y_train = tf.cast(y_train, dtype=tf.float64)
    loss = tf.keras.losses.CategoricalCrossentropy()
    # Use Keras to define our model
    WIDTH = 125
    model = Sequential()
    model.add(Dense(WIDTH, activation="relu", input_shape=(1, numInputs)))
    model.add(Dense(numOut, activation="softmax"))
    print("Ins and Outs: ", numInputs, numOut)
    # Use deepbayesHF to define our optimizer
    optimizer = optimizers.PriorHamiltonianMonteCarlo()
    bayes_model = optimizer.compile(model, loss_fn=loss, learning_rate = 0.1, mh_burn=False,
                          epochs=25, decay=0.3, steps = 30, b_steps = 25, burn_in=20,
                          rob_lam=0.8, classes = 5, path = "Posteriors/HMC_HCAS_BNN_%s_%s"%(pra, tau),
                          inflate_prior=500., mode='classification') # select optimizer and set learning rate

    # Train and save BNN using deepbayesHF
    #bayes_model.train(np.asarray([X_train]), y_train, np.asarray([X_train]), np.asarray([y_train]))
    bayes_model.constraint_train(np.asarray([X_train]), np.asarray([y_train]), 
                            np.asarray([X_train])-0.012, np.asarray([X_train])+0.012,
                            np.asarray([X_train]), np.asarray([y_train]))
    bayes_model.finalize_chain()
    y_pred = bayes_model.predict(X_train)

    #bayes_model.save("Posteriors/ROB_HCAS_BNN_%s_%s"%(pra, tau))
    bayes_model.save("Posteriors/HMC_HCAS_BNN_%s_%s"%(pra, tau))
