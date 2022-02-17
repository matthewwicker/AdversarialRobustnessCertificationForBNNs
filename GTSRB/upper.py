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
from deepbayesHF.analyzers import decision_veri_upper
from deepbayesHF.analyzers import FGSM

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum", default=0.0)
parser.add_argument("--eps", default=0.0)
parser.add_argument("--lam", default=1.0)
parser.add_argument("--rob", default=0)
parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
parser.add_argument("--opt")
parser.add_argument("--width", default=24)
parser.add_argument("--depth", default=1)
parser.add_argument("--data")


args = parser.parse_args()
imnum = int(args.imnum)
eps = float(args.eps)
lam = float(args.lam)
optim = str(args.opt)
rob = int(args.rob)
width = int(args.width)
depth = int(args.depth)
post_string = str(args.opt)
benchmark = str(args.data)
INDEX = imnum


EPSILON = 1/255
MARGIN = 4.0
SAMPLES = 3
MAXDEPTH = 3

# 2.5, 750
# LOAD IN THE DATA

#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#X_train = X_train/255.
#X_test = X_test/255.
#X_train = X_train.astype("float64").reshape(-1, 28*28)
#X_test = X_test.astype("float64").reshape(-1, 28* 28)

#X_train = np.load("data/xtrain.npy").astype("float32") + 0.5
#y_train = np.load("data/ytrain.npy")
X_test = np.load("data/xtest.npy").astype("float32") + 0.5
#y_test = np.load("data/ytest.npy")
#y_train = np.argmax(y_train, axis=1)
#y_test = np.argmax(y_test, axis=1)

from PIL import Image

if(benchmark == 'gtsrb'):
    images = X_test
else:
    benchmark = 'OOD/' + benchmark
    image_paths = [f for f in os.listdir(benchmark) if os.path.isfile(os.path.join(benchmark, f))]
    images = []
    # Open the image form working directory
    for i in image_paths:
        image = Image.open(benchmark+'/'+i)
        # summarize some details about the image
        image = np.asarray(image)
        image = image[:, : ,0:3]
        images.append(image)

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
    worst_case = tf.math.add(tf.math.multiply(v1, ou), tf.math.multiply(v2, ol))
    worst_case = tf.nn.softmax(worst_case)
    return worst_case[TRUE_VALUE]


import numpy as np
# Load in approximate posterior distribution
#bayes_model = PosteriorModel("Posteriors/%s_FCN_Posterior_%s_%s_%s_%s_%s"%(optim, width, depth, rob, lam, eps))
bayes_model = PosteriorModel("Posteriors/VOGN_small_Posterior_5")
bayes_model.posterior_var += 0.000000001 # #nsuring 0s get rounded up to small values

image = images[imnum]
# SELECT THE INPUT
img = np.asarray(image)
img = img.astype('float32')
y_pred = bayes_model.predict(np.asarray([img]))
TRUE_VALUE = np.argmax(y_pred) #y_test[INDEX]

import json
dir = "ExperimentalLogs"
post_string = "%s_FCN_%s_%s_%s_%s_%s_upper.log"%(optim, width, depth, rob, lam, eps)

EPSILON = 1/255
img_upper = np.clip(np.asarray([img+(EPSILON)]), 0, 1)
img_lower = np.clip(np.asarray([img-(EPSILON)]), 0, 1)
p_lower = decision_veri_upper(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, depth=MAXDEPTH, value=logit_value)
print("Computed Upper Bounds", p_lower)

iterations = 0
record = {"Index":INDEX, "Upper":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "MaxEps":EPSILON,  "Samples":SAMPLES, "Depth":MAXDEPTH, "Data":benchmark}

with open("%s/%s"%(dir, post_string), 'a') as f:
    json.dump(record, f)
    f.write(os.linesep)



