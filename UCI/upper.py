# Author: Matthew Wicker

# Alright, hold on to your socks this one might be more rough...
import numpy as np
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

sys.path.append('../')
import deepbayesHF
import deepbayesHF.optimizers as optimizers
from deepbayesHF.analyzers import prob_veri_upper as IBP_prob

# All of the inputs are basically the same except for this
from deepbayesHF import PosteriorModel
#from deepbayesHF.analyzers import IBP_prob
#from deepbayesHF.analyzers import IBP_upper
from deepbayesHF.analyzers import FGSM
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from datasets import Dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--dataset")

args = parser.parse_args()
imnum = int(args.imnum)
dataset_string = str(args.dataset)
post_string = dataset_string
INDEX = imnum

# What should make the upper bound (eps) lower:
#	*) a lower delta

# VERIFICATION HYPER PARAMETERS
#MARGIN = 2.0
#SAMPLES = 10
#MAXDEPTH = 5
MARGIN = 2.25
SAMPLES = 3
MAXDEPTH = 3

data = Dataset(dataset_string)
from sklearn.preprocessing import StandardScaler

X_train = np.asarray(data.train_set.train_data)
y_train = np.asarray(data.train_set.train_labels.reshape(-1,1))

X_test = np.asarray(data.test_set.test_data)
y_test = np.asarray(data.test_set.test_labels.reshape(-1,1))

X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train, X_test = X_scaler.transform(X_train), X_scaler.transform(X_test)

# If we are running with scaled posteriors
y_scaler = StandardScaler()
y_scaler.fit(y_train)
y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)


print("Sucessfully loaded dataset %s \t \t with train shapes %s, %s"%(dataset_string, X_train.shape, y_train.shape))

input_range = (np.max(X_test, axis=0) - np.min(X_test, axis=0))/2
print("Input range: ", input_range)
output_range = np.max(y_test) - np.min(y_test)
print("Output range: ", np.max(y_test) - np.min(y_test))
DELTA = 0.25 * (output_range)
print("HERE IS DELTA: ", DELTA)

in_dims = X_train.shape[1]

# Dataset information:
in_dims = X_train.shape[1]

import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("IEEEPosterior/VOGN_%s_Model"%(dataset_string))

# Lets define our correctness property (e.g. correct class)
TRUE_VALUE = y_test[INDEX]

def predicate_unsafe(iml, imu, ol, ou):
    bound_above = TRUE_VALUE + DELTA
    bound_below = TRUE_VALUE - DELTA
    #print(bound_below,bound_above)
    #print(ol,ou)
    if(ou <= bound_below or ol >= bound_above):
        return True
    else:
        return False

import json
dir = "Logs"

# ===================================================
#        Check down coarse grained
# ===================================================
DONE = False
for eps in np.linspace(1.0, 0.1, 19):
    img = np.asarray([X_test[INDEX]])
    img_upper = np.asarray([X_test[INDEX]+(input_range*eps)])
    img_lower = np.asarray([X_test[INDEX]-(input_range*eps)])
    # We start with epsilon = 0.0 and increase it as we go.
    p_upper = IBP_prob(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_unsafe, depth=MAXDEPTH)
    p_upper = 1-p_upper
    print("~~~~~~~~~~~~~~~~~~~~~~~~ Safety Probability: ", p_upper)
    if(p_upper > 0.75):
        DONE = True
        break

if(DONE):
    iterations = 0
    record = {"Index":INDEX, "Lower":p_upper, "Samples":SAMPLES, "Margin":MARGIN, "MaxEps":eps,  "Samples":SAMPLES, "Depth":MAXDEPTH}
    with open("%s/%s_upper.log"%(dir, post_string), 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)
        sys.exit(0)

# ===================================================
#        Then we systematically check down
# ===================================================
for eps in np.linspace(0.1, 0.01, 10):
    img = np.asarray([X_test[INDEX]])
    img_upper = np.asarray([X_test[INDEX]+(input_range*eps)])
    img_lower = np.asarray([X_test[INDEX]-(input_range*eps)])
    # We start with epsilon = 0.0 and increase it as we go.
    p_upper = IBP_prob(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_unsafe, depth=MAXDEPTH)
    p_upper = 1-p_upper
    print("~~~~~~~~~~~~~~~~~~~~~~~~ Safety Probability: ", p_upper)
    if(p_upper > 0.75):
        break
print("Radius: ", eps)

import json
iterations = 0
record = {"Index":INDEX, "Lower":p_upper, "Samples":SAMPLES, "Margin":MARGIN, "MaxEps":eps,  "Samples":SAMPLES, "Depth":MAXDEPTH}
with open("%s/%s_upper.log"%(dir, post_string), 'a') as f:
    json.dump(record, f)
    f.write(os.linesep)
    sys.exit(0)

