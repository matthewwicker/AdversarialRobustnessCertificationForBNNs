import numpy as np
import sys, os
sys.path.append("../../")
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=0.004)
parser.add_argument("--lam", default=0.25)
parser.add_argument("--rob", default=5)
parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')
parser.add_argument("--opt")

args = parser.parse_args()
eps = float(args.eps)
lam = float(args.lam)
rob = int(args.rob)
optim = str(args.opt)
gpu = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import deepbayesHF
import deepbayesHF.optimizers as optimizers

X_train = np.load("data/xtrain.npy").astype("float32") + 0.5
y_train = np.load("data/ytrain.npy")
X_test = np.load("data/xtest.npy").astype("float32") + 0.5
y_test = np.load("data/ytest.npy")

"""
augment_size = 5000
image_generator = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.075, 
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=False, 
    data_format="channels_last",
    zca_whitening=False)
# fit data for zca whitening
image_generator.fit(X_train, augment=True)
randidx = np.random.randint(3000, size=augment_size)
x_augmented = X_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                            batch_size=augment_size, shuffle=False).next()[0]
# append augmented data to trainset
X_train = np.concatenate((X_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented)) * 1.1
"""

#y_train = np.argmax(y_train, axis=1)
#y_test = np.argmax(y_test, axis=1)



print("SHAPES -------", X_train.shape, y_train.shape)

if(True):
    model = Sequential()
    model.add(Conv2D(filters=1, kernel_size=(4, 4), activation='relu', input_shape=(28,28,3)))
    #model.add(Conv2D(filters=10, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Flatten())
    model.add(Dense(25, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))

print(model.summary())

"""
batch_size = 128
epochs = 15
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Here we are trying a cheap swag approximation because BayesKeras is messing up.
params = []
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
w = model.get_weights(); params.append(w)

model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
w = model.get_weights(); params.append(w)

opt = optimizers.StochasticWeightAveragingGaussian()
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model_type = "small"
bayes_model = opt.compile(model, loss_fn=loss)
bayes_model.weights_stack = params
bayes_model.get_posterior()
bayes_model.save("Posteriors/%s_%s_Posterior_%s_new"%(optim, model_type, rob))
sys.exit(0)
"""

lr = 1
print("Got flag: %s"%(optim))
if(optim == 'VOGN'):
#    learning_rate = 0.35*lr; decay=0.075
    learning_rate = 0.01*lr; decay=0.0
    #learning_rate = 0.05*lr; decay=0.0
    opt = optimizers.VariationalOnlineGuassNewton()
elif(optim == 'BBB'):
    learning_rate = 0.65*lr; decay=0.0
    opt = optimizers.BayesByBackprop()
elif(optim == 'SWAG'):
#    learning_rate = 0.0125*lr; decay=0.025
    learning_rate = 0.015*lr; decay=0.0
    opt = optimizers.StochasticWeightAveragingGaussian()
elif(optim == 'NA'):
    learning_rate = 0.00005*lr; decay=0.025
    opt = optimizers.NoisyAdam()
elif(optim == 'SGD'):
    learning_rate = 0.05*lr; decay=0.1
    opt = optimizers.StochasticGradientDescent()
# Compile the model to train with Bayesian inference
if(rob == 0 or rob == 3 or rob == 4 or rob == 5):
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = tf.keras.losses.CategoricalCrossentropy()
else:
    loss = BayesKeras.optimizers.losses.robust_crossentropy_loss


inf = 0.01
model_type = "small"
#learning_rate *= 1.5
bayes_model = opt.compile(model, loss_fn=loss, epochs=30, learning_rate=learning_rate, batch_size=128, input_noise=0.0,
                          decay=decay, robust_train=rob, epsilon=eps, rob_lam=lam, inflate_prior=inf, classes=2,
                          log_path="Posteriors/%s_%s_Posterior_%s_new.log"%(optim, model_type, rob))

# Train the model on your data
bayes_model.train(X_train, y_train, X_test, y_test)

# Save your approxiate Bayesian posterior
bayes_model.save("Posteriors/%s_%s_Posterior_%s_new"%(optim, model_type, rob))





