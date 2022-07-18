# Author: Matthew Wicker

# Alright, hold on to your socks this one might be more rough...
import numpy as np
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

sys.path.append('../')
import deepbayesHF
import deepbayesHF.optimizers as optimizers
from deepbayesHF.analyzers import prob_veri as IBP_prob

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

# VERIFICATION HYPER PARAMETERS
MARGIN = 2.25
SAMPLES = 5
MAXDEPTH = 5

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

# Dataset information:
in_dims = X_train.shape[1]

# Dataset information:
in_dims = X_train.shape[1]

import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("IEEEPosterior/VOGN_%s_Model"%(dataset_string))

# Lets define our correctness property (e.g. correct class)
TRUE_VALUE = y_test[INDEX]
EPSILON = 0.015

def predicate_safe(iml, imu, ol, ou):
    bound_above = TRUE_VALUE + DELTA
    bound_below = TRUE_VALUE - DELTA
    #print(bound_below, bound_above)
    #print(ol, ou, "subset?", bound_below, bound_above)
    if(ol >= bound_below and ou <= bound_above):
        return True
    else:
        return False

#import time
import json
dir = "Logs"
for eps in np.linspace(0.01, 0.2, 16):
    img = np.asarray([X_test[INDEX]])
    img_upper = np.asarray([X_test[INDEX]+(input_range*eps)])
    img_lower = np.asarray([X_test[INDEX]-(input_range*eps)])
    # We start with epsilon = 0.0 and increase it as we go.
    #start = time.time()
    p_lower = IBP_prob(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, depth=MAXDEPTH)
    #end = time.time()
    #break
    print("~~~~~~~~~ Safety Probability: ", p_lower)
    if(p_lower < 0.5):
        break
eps -= 0.01
print("Radius: ", eps)
#print("Time: ", end - start)
#iterations = 0
#record = {"Index":INDEX, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "MaxEps":eps,  "Samples":SAMPLES, "Depth":MAXDEPTH}
#with open("%s/%s_lower.log"%(dir, post_string), 'a') as f:
#    json.dump(record, f)
#    f.write(os.linesep)

