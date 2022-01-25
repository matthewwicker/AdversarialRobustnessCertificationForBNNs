# Author: Matthew Wicker

# Alright, hold on to your socks this one might be more rough...
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import sys
import logging
sys.path.append("../")
import deepbayesHF
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import decision_veri
from deepbayesHF.analyzers import FGSM

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum", default=0)
parser.add_argument("--eps", default=0.05)
parser.add_argument("--lam", default=0.0)
parser.add_argument("--rob", default=5)
parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
parser.add_argument("--opt", default="VOGN")
parser.add_argument("--width", default=32)
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


EPSILON = 0.05
MARGIN = 3.1
SAMPLES = 3
MAXDEPTH = 3

# 2.5, 750
# LOAD IN THE DATA

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float64").reshape(-1, 28*28)
X_test = X_test.astype("float64").reshape(-1, 28* 28)

def predicate_safe(iml, imu, ol, ou):
    v1 = tf.one_hot(TRUE_VALUE, depth=10)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=10)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    if(np.argmax(worst_case) == TRUE_VALUE):
        return True
    else:
        return False

def logit_value(iml, imu, ol, ou):
    v1 = tf.one_hot(TRUE_VALUE, depth=10)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=10)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    worst_case = tf.nn.softmax(worst_case)
    return worst_case[TRUE_VALUE]
    #if(np.argmax(worst_case) == TRUE_VALUE):
    #    return True
    #else:
    #    return False



import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("Posteriors/%s_FCN_Posterior_%s_%s_%s_%s_%s"%(optim, width, depth, rob, lam, eps))
bayes_model.posterior_var += 0.000000001 # #nsuring 0s get rounded up to small values

# SELECT THE INPUT
img = np.asarray([X_test[INDEX]])
TRUE_VALUE = np.argmax(bayes_model.predict(np.asarray([img]))) #y_test[INDEX]

import json
dir = "DecLogs"
post_string = "%s_FCN_%s_%s_%s_%s_%s_lower.log"%(optim, width, depth, rob, lam, eps)

for EPSILON in np.linspace(0.01, 0.25, 16):
    img = np.asarray([X_test[INDEX]])
    img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
    img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)
    p_lower = decision_veri(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, value=logit_value, depth=MAXDEPTH)
    record = {"Index":INDEX, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "MaxEps":EPSILON,  "Samples":SAMPLES, "Depth":MAXDEPTH, "layers":depth}
    with open("%s/%s"%(dir, post_string), 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)
    print("~~~~~~~~~ Decision Probability: ", p_lower)




