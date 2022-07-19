# Author: Matthew Wicker

# Alright, hold on to your socks this one might be more rough...
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import sys
import logging
sys.path.append("../")
import deepbayesHF
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import prob_veri
from deepbayesHF.analyzers import decision_veri
from deepbayesHF.analyzers import FGSM

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np
import argparse
RUN_INDEX = 4
parser = argparse.ArgumentParser()
parser.add_argument("--imnum", default=0.0)
parser.add_argument("--eps", default=0.0)
parser.add_argument("--lam", default=1.0)
parser.add_argument("--rob", default=0)
parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
parser.add_argument("--opt")
parser.add_argument("--width", default=24)
parser.add_argument("--depth", default=1)


args = parser.parse_args()
imnum = int(args.imnum)
eps = float(args.eps)
lam = float(args.lam)
optim = str(args.opt)
rob = int(args.rob)
width = int(args.width)
depth = int(args.depth)
post_string = str(args.opt)
INDEX = imnum


EPSILON = 1/255
MARGIN = 3.85
SAMPLES = 1
MAXDEPTH = 1

# 2.5, 750
# LOAD IN THE DATA

#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#X_train = X_train/255.
#X_test = X_test/255.
#X_train = X_train.astype("float64").reshape(-1, 28*28)
#X_test = X_test.astype("float64").reshape(-1, 28* 28)

X_train = np.load("data/xtrain.npy").astype("float32") + 0.5
y_train = np.load("data/ytrain.npy")
X_test = np.load("data/xtest.npy").astype("float32") + 0.5
y_test = np.load("data/ytest.npy")
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

def predicate_safe(iml, imu, ol, ou):
    return True
    v1 = tf.one_hot(TRUE_VALUE, depth=2)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=2)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    if(np.argmax(worst_case) == TRUE_VALUE):
        return True
    else:
        return False


def logit_value(iml, imu, ol, ou):
    v1 = tf.one_hot(TRUE_VALUE, depth=2)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=2)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    worst_case = tf.nn.softmax(worst_case)
    return worst_case[TRUE_VALUE]


import numpy as np
# Load in approximate posterior distribution
#bayes_model = PosteriorModel("Posteriors/%s_FCN_Posterior_%s_%s_%s_%s_%s"%(optim, width, depth, rob, lam, eps))
bayes_model = PosteriorModel("Posteriors/VOGN_small_Posterior_5_new")
bayes_model.posterior_var += 0.000000001 # #nsuring 0s get rounded up to small values

for INDEX in range(RUN_INDEX*50,(RUN_INDEX+1)*50):
    # SELECT THE INPUT
    img = np.asarray([X_test[INDEX]])

    #print("OBSERVE: ")
    TRUE_VALUE = y_test[INDEX]
    #y_pred = bayes_model.predict(img)
    #print(y_pred)
    y_pred = np.argmax(bayes_model.predict(img))
    #print(TRUE_VALUE, y_pred)
    TRUE_VALUE = y_pred
    #print(TRUE_VALUE)
    #print(y_pred, TRUE_VALUE)
    #continue
    import json
    dir = "ExperimentalLogs"
    post_string = "PAPER_lower.log" #%(optim, width, depth, rob, lam, eps)

    #for EPSILON in np.linspace(0.0, 0.5, 26):
    #EPSILON = 1/255
    img = np.asarray([X_test[INDEX]])
    img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
    img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)
    import time
    start = time.time()
    p_lower = decision_veri(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, depth=MAXDEPTH, value=logit_value)
    end = time.time()
    print("TIME: ", end - start)
    #record = {"Index":INDEX, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "MaxEps":EPSILON,  "Samples":SAMPLES, "Depth":MAXDEPTH}
    #with open("%s/%s"%(dir, post_string), 'a') as f:
    #    json.dump(record, f)
    #    f.write(os.linesep)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)
    print("PROB: ", p_lower)


