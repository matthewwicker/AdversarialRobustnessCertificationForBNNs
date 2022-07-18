# Train All Posteriors for MNIST Evaluation


# Author: Matthew Wicker

# Description: Minimal working example of training and saving
# a BNN trained with Bayes by backprop (BBB)
# can handle any Keras model
import sys, os
from pathlib import Path
path = Path(os.getcwd())
#sys.path.append(str(path.parent))

sys.path.append("../")
import deepbayesHF
import deepbayesHF.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

#tf.debugging.set_log_device_placement(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=0.0)
parser.add_argument("--lam", default=1.0)
parser.add_argument("--rob", default=0)
parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
parser.add_argument("--opt")
parser.add_argument("--width", default=24)
parser.add_argument("--depth", default=1)

args = parser.parse_args()
eps = float(args.eps)
lam = float(args.lam)
optim = str(args.opt)
rob = int(args.rob)
width = int(args.width)
depth = int(args.depth)
gpu = str(args.gpu)
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28* 28)

#X_train = X_train[0:10000]
#y_train = y_train[0:10000]
model = Sequential()
for i in range(depth):
    if(i == 0):
        model.add(Dense(width, activation="relu", input_shape=(None, 28*28)))
    else:
        model.add(Dense(width, activation="relu"))
model.add(Dense(10, activation="softmax"))

inf = 2
full_covar = False
if(optim == 'VOGN'):
    inf = 2
    learning_rate = 0.35; decay=0.0
    opt = optimizers.VariationalOnlineGuassNewton()
elif(optim == 'BBB'):
    inf = 10
    learning_rate = 0.15; decay=0.0
    opt = optimizers.BayesByBackprop()
elif(optim == 'SWAG'):
    learning_rate = 0.01; decay=0.0
    opt = optimizers.StochasticWeightAveragingGaussian()
elif(optim == 'SWAG-FC'):
    learning_rate = 0.01; decay=0.0; full_covar=True
    opt = optimizers.StochasticWeightAveragingGaussian()
elif(optim == 'SGD'):
    learning_rate = 1.0; decay=0.0
    opt = optimizers.StochasticGradientDescent()
elif(optim == 'NA'):
    inf = 2
    learning_rate = 0.001; decay=0.0
    opt = optimizers.NoisyAdam()
elif(optim == 'ADAM'):
    learning_rate = 0.00001; decay=0.0
    opt = optimizers.Adam()
elif(optim == 'HMC'):
    learning_rate = 0.075; decay=0.0; inf=250
    linear_schedule = False
    opt = optimizers.HamiltonianMonteCarlo()

# Compile the model to train with Bayesian inference
#if(rob == 0):
loss = tf.keras.losses.SparseCategoricalCrossentropy()
#elif(rob != 0):
#    loss = deepbayesHF.optimizers.losses.robust_crossentropy_loss

bayes_model = opt.compile(model, loss_fn=loss, epochs=35, learning_rate=learning_rate,
                          batch_size=128, linear_schedule=True,
                          decay=decay, robust_train=rob, inflate_prior=inf,
                          burn_in=3, steps=25, b_steps=20, epsilon=eps, rob_lam=lam) #, preload="SGD_FCN_Posterior_1")
#steps was 50
# Train the model on your data
bayes_model.train(X_train, y_train, X_test, y_test)

# Save your approxiate Bayesian posterior
bayes_model.save("Posteriors/%s_FCN_Posterior_%s_%s_%s_%s_%s"%(optim, width, depth, rob, lam, eps))
