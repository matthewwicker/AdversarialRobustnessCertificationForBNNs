#HMC = False
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
    p_concur = int(sys.argv[4])
    n_concur = int(sys.argv[5])
    HMC = True #bool(int(sys.argv[6]))
    if(HMC):
        from deepbayesHF.analyzers import prob_veri_samp as prob_veri
        from deepbayesHF.analyzers import decision_veri_samp as decision_veri 
        from deepbayesHF.analyzers import decision_veri_samp as decision_veri_upper 
    else:
        from deepbayesHF.analyzers import prob_veri
        from deepbayesHF.analyzers import decision_veri  

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
        return True
        #v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
        #v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
        #v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        #worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
        #if(np.argmax(worst_case) not in NOTIN):
        #    return True
        #else:
        #    return False

    def logit_value(iml, imu, ol, ou):
        v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
        v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
        v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
        worst_case = tf.nn.softmax(worst_case)
        return worst_case[TRUE_VALUE]

    def phi_n_upper(iml, imu, ol, ou):
        return True


    def logit_value_upper(iml, imu, ol, ou):
        v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
        v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
        v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        best_case = tf.math.add(tf.math.multiply(v1, ou), tf.math.multiply(v2, ol))
        best_case = tf.nn.softmax(best_case)
        return best_case[TRUE_VALUE]



    print("Setting up Verification")
    if(HMC):
        print("LOADING HMC")
        bayes_model = PosteriorModel("Posteriors/HMC_HCAS_BNN_%s_%s"%(pra, tau))
    else:
        bayes_model = PosteriorModel("Posteriors/ROB_HCAS_BNN_%s_%s"%(pra, tau))
    mins = np.min(X_train, axis=0)
    maxs = np.max(X_train, axis=0)
    desc = [25,5,5]
    if(HMC):
        radi = ((maxs-mins)/desc)/10
    else:
        radi = ((maxs-mins)/desc)/10
    inps = []
    lowers = []
    uppers = []
    for a in np.linspace(mins[0], maxs[0], desc[0]):
        for b in np.linspace(mins[1], maxs[1], desc[1]):
            for c in np.linspace(mins[2], maxs[2], desc[2]):
                inps.append(np.asarray([a,b,c]))
                lowers.append(np.asarray([a,b,c] - radi))
                uppers.append(np.asarray([a,b,c] + radi))

    indexes = np.linspace(0, len(inps), n_concur)
    a = int(indexes[p_concur])
    b = int(indexes[p_concur+1])
    inps = inps[a:b]
    y_pred = bayes_model.predict(np.asarray([inps]))
    y_pred = np.squeeze(y_pred)
    classes = np.argmax(y_pred, axis=1)
    indforprop = []

    for i in range(len(inps)):
        print(classes[i], CLS_VALUES)
        if(classes[i] in CLS_VALUES):
            indforprop.append(i)


    for i in indforprop:
        TRUE_VALUE = classes[i]
        MAXDEPTH = 3
        SAMPLES = 3
        MARGIN = 2.75

        inp_upper = np.asarray([uppers[i]])
        inp_lower = np.asarray([lowers[i]])

        p_lower = decision_veri(bayes_model, inp_lower, inp_upper, MARGIN, SAMPLES, predicate=phi_n, depth=MAXDEPTH, value=logit_value)
        p_lower = float(p_lower)
        print("Initial Safety Probability: ", p_lower)

        #Save result to log files
        record = {"Index":a+i, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":-1, "Depth":MAXDEPTH, "PRA":pra, "TAI":tau, "PHI":phi}
        if(HMC):
            post_string = "NHB_Bounds_%s_%s_%s"%(pra, tau, phi)
        else:
            post_string = "NVB_Bounds_%s_%s_%s"%(pra, tau, phi)
        #for i in range(10):
        print(post_string, record)
        with open("%s/%s_lower.log"%("CompareLogs", post_string), 'a') as f:
            json.dump(record, f)
            f.write(os.linesep)


        p_upper = decision_veri_upper(bayes_model, inp_lower, inp_upper, MARGIN, SAMPLES, 
                                      predicate=phi_n_upper, depth=MAXDEPTH, value=logit_value_upper)
        p_upper = float(p_upper)
        print("Initial Safety Probability: ", p_upper)

        #Save result to log files
        record = {"Index":a+i, "Upper":p_upper, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":-1, "Depth":MAXDEPTH, "PRA":pra, "TAI":tau, "PHI":phi}
        if(HMC):
            post_string = "NHB_Bounds_%s_%s_%s"%(pra, tau, phi)
        else:
            post_string = "NVB_Bounds_%s_%s_%s"%(pra, tau, phi)
        #for i in range(10):
        print(post_string, record)
        with open("%s/%s_lower.log"%("CompareLogs", post_string), 'a') as f:
            json.dump(record, f)
            f.write(os.linesep)


