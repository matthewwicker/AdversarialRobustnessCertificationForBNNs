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
from deepbayesHF.analyzers import FGSM

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--infer")
parser.add_argument("--width")
parser.add_argument("--depth")

args = parser.parse_args()
imnum = int(args.imnum)
width = int(args.width)
depth = int(args.depth)
post_string = str(args.infer)
INDEX = imnum


EPSILON = 0.025
MARGIN = 3.0
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


import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("Posteriors/%s_FCN_Posterior_%s_%s"%(post_string, width, depth))

bayes_model.posterior_var += 0.000000001
# SELECT THE INPUT
img = np.asarray([X_test[INDEX]])
#TRUE_VALUE = y_test[INDEX]
TRUE_VALUE = np.argmax(bayes_model.predict(np.asarray([img]))) #y_test[INDEX]


img = np.asarray([X_test[INDEX]])
img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)


# We start with epsilon = 0.0 and increase it as we go.
p_lower = prob_veri(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, depth=MAXDEPTH)
print("Initial Safety Probability: ", p_lower)
#dir = "ExperimentalLogsVOGN"
#import json
#iterations = 0
#record = {"Index":INDEX, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":EPSILON,  "Iterations":iterations, "Width":width, "Depth":depth}
#with open("%s/%s_lower.log"%(dir, post_string), 'a') as f:
#    json.dump(record, f)
#    f.write(os.linesep)

