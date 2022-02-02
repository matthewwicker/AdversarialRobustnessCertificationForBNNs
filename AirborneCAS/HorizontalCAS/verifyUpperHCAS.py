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
from deepbayesHF.analyzers import prob_veri_upper


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


# The previous RA should be given as a command line input
if len(sys.argv) > 2:
    pra = int(sys.argv[1])
    tau = int(sys.argv[2])
    phi = int(sys.argv[3])
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

    if(phi < 5):
        TRUE_VALUE = phi
        CLS_VALUES = [phi]
        NOTIN = [0,1,2,3,4]
        del NOTIN[phi]

    elif(phi == 5):
        CLS_VALUES = [3,4]
        NOTIN = [1,2]

    elif(phi == 6):
        CLS_VALUES = [1,2]
        NOTIN = [3,4]

    elif(phi == 7):
        CLS_VALUES = [1,4]
        NOTIN = [0]



    def process_log(path_to_log):
        rows = []
        indexes = []
        with open(path_to_log) as log_file:
            for line in log_file:
                #print(json.loads(line))
                entry = json.loads(line)
                if(float(entry["Lower"]) < 0.9):
                    indexes.append(int(entry["Index"]))
                rows.append(json.loads(line))
        return indexes #pd.DataFrame.from_dict(rows, orient='columns')

    path = "LogFiles/HCAS_Bounds_%s_%s_%s_lower.log"%(pra, tau, phi)
    inputs = process_log(path)
    #classes = tf.argmax(y_train,axis=1)
    #inputs = []
    #for i in range(len(y_train)):
    #    print(classes[i], CLS_VALUES)
    #    if(classes[i] in CLS_VALUES):
    #        inputs.append(i)
    #y_train = classes
    y_train = tf.argmax(y_train, axis=1)
    classes = y_train

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
        best_case = tf.math.add(tf.math.multiply(v1, ou), tf.math.multiply(v2, ol))
        #worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
        if(np.argmax(best_case) in NOTIN):
            return True
        else:
            return False

    print("Setting up Verification")
    print("Verifying %s of %s inputs"%(len(inputs), len(y_train)))
    bayes_model = PosteriorModel("Posteriors/ROB_HCAS_BNN_%s_%s"%(pra, tau))

    for i in range(len(inputs)):
        INDEX = inputs[i]
        TRUE_VALUE = classes[INDEX]
        EPSILON = 0.025
        MAXDEPTH = 3
        SAMPLES = 3
        MARGIN = 2.5

        y_pred = bayes_model.predict(np.asarray([X_train[INDEX]]))
        #if(np.argmax(y_pred) != TRUE_VALUE):
        #    print("misclassified")
        #    record = {"Index":INDEX, "Lower":0.0, "Samples":0, "Margin":MARGIN, "Epsilon":EPSILON, "Depth":MAXDEPTH, "PRA":pra, "TAI":tau, "PHI":phi}
        #    post_string = "HCAS_Bounds_%s_%s_%s"%(pra, tau, phi)
        #    with open("%s/%s_upper.log"%("LogFiles", post_string), 'a') as f:
        #        json.dump(record, f)
        #        f.write(os.linesep)
        #    continue

        inp_upper = np.asarray([X_train[INDEX]+(EPSILON)]) #np.clip(np.asarray(X_train[INDEX]+(EPSILON)), -100000, 100000)
        inp_lower = np.asarray([X_train[INDEX]-(EPSILON)]) #np.clip(np.asarray(X_train[INDEX]-(EPSILON)), -100000, 100000)

        p_lower = prob_veri_upper(bayes_model, inp_lower, inp_upper, MARGIN, SAMPLES, predicate=phi_n, depth=MAXDEPTH)
        p_lower = 1-p_lower
        print("Initial Upper Bound On Safety Probability: ", p_lower)

        #Save result to log files
        record = {"Index":INDEX, "Upper":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":EPSILON, "Depth":MAXDEPTH, "PRA":pra, "TAI":tau, "PHI":phi}
        post_string = "HCAS_Bounds_%s_%s_%s"%(pra, tau, phi)
        with open("%s/%s_upper.log"%("LogFiles", post_string), 'a') as f:
            json.dump(record, f)
            f.write(os.linesep)


