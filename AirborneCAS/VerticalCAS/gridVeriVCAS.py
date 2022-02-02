import numpy as np
import math
import tensorflow as tf
import h5py
import json
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

    phi = int(sys.argv[2])
    p_concur = int(sys.argv[3])
    n_concur = int(sys.argv[4])
    gpu = -1
    print("Verifying the following: ")
    print("Posterior Path: \t Posteriors/NEW_VCAS_BNN_%s"%(pra))
    print("Log path: \t \t ")
    print("Consistency property: %s"%(property))

    N,numInputs = X_train.shape
    N,numOut = Q.shape
    phi = phi - 1

    if(phi < 10):
        TRUE_VALUE = phi
        CLS_VALUES = [phi]
        NOTIN = [0,1,2,3,4,5,6,7,8]
        del NOTIN[phi]

    # These have not been tuned to VCAS
    elif(phi == 10):
        CLS_VALUES = [3,4]
        NOTIN = [1,2]

    elif(phi == 11):
        CLS_VALUES = [1,2]
        NOTIN = [3,4]

    elif(phi == 12):
        CLS_VALUES = [1,4]
        NOTIN = [0]


    def phi_n(iml, imu, ol, ou):
        """
        * Phi_n - general function just specifying what the output cannot be
        -> phi_0 - notin=[1,2,3,4]          INPUTS: class=[0]
        -> phi_1 - notin=[0,2,3,4]          INPUTS: class=[1]
        -> phi_2 - notin=[1,0,3,4]          INPUTS: class=[2]
        -> phi_3 - notin=[1,2,0,4]          INPUTS: class=[3]
        -> phi_4 - notin=[1,2,3,0]	    INPUTS: class=[4]
        -> phi_5 - notin=[1,2]		    INPUTS: class=[3,4]
        -> phi_6 - notin=[3,4]		    INPUTS: class=[1,2]
        -> phi_7 - notin=[0]		    INPUTS: class=[1,4]
        """
        v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
        v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
        v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
        if(np.argmax(worst_case) not in NOTIN):
            return True
        else:
            return False

    print("Setting up Verification")
    bayes_model = PosteriorModel("Posteriors/NEW_VCAS_BNN_%s"%(pra))

    mins = np.min(X_train, axis=0)
    maxs = np.max(X_train, axis=0)
    desc = [15,10,10,5]
    radi = ((maxs-mins)/desc)/4

    inps = []
    lowers = []
    uppers = []
    print((maxs[0]-mins[0])/25)
    for a in np.linspace(mins[0], maxs[0], desc[0]):
        for b in np.linspace(mins[1], maxs[1], desc[1]):
            for c in np.linspace(mins[2], maxs[2], desc[2]):
                for d in np.linspace(mins[3], maxs[3], desc[3]):
                        inps.append([a,b,c,d])
                        lowers.append([a,b,c,d] - radi)
                        uppers.append([a,b,c,d] + radi)


    indexes = np.linspace(0, len(inps), n_concur)
    a = int(indexes[p_concur])
    b = int(indexes[p_concur+1])
    inps = inps[a:b]
    y_pred = bayes_model.predict(np.asarray([inps]))
    #y_pred = bayes_model.predict(X_train)
    y_pred = np.squeeze(y_pred)
    classes = np.argmax(y_pred, axis=1)
    for i in classes:
        print(i)
        if(i != 0):
            print(i)
    sys.exit(0)
    indforprop = []
    for i in range(len(y_train)):
        if(np.argmax(y_train[i]) != 0):
            y_p = bayes_model.predict(np.asarray([X_train[i]]))
            print(np.argmax(y_train[i]), np.argmax(y_p)) 
    print(np.argmax(y_train, axis=1))
    print(classes)
    sys.exit(0)

    for i in range(len(inps)):
        #print(classes[i], CLS_VALUES)
        if(classes[i] in CLS_VALUES):
            indforprop.append(i)
    print("Found this many for verification: ", len(indforprop))

    for i in indforprop:
        TRUE_VALUE = classes[i]
        MAXDEPTH = 3
        SAMPLES = 3
        MARGIN = 2.75

        inp_upper = np.asarray([uppers[i]])
        inp_lower = np.asarray([lowers[i]])

        p_lower = prob_veri(bayes_model, inp_lower, inp_upper, MARGIN, SAMPLES, predicate=phi_n, depth=MAXDEPTH)
        print("Initial Safety Probability: ", p_lower)

        #Save result to log files
        record = {"Index":a+i, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":-1, "Depth":MAXDEPTH, "PRA":pra, "TAI":tau, "PHI":phi}
        post_string = "HCAS_Bounds_%s_%s_%s"%(pra, tau, phi)
        with open("%s/%s_lower.log"%("GridLogs", post_string), 'a') as f:
            json.dump(record, f)
            f.write(os.linesep)


