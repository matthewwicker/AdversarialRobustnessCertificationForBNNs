
import numpy as np
import sys, os
from pathlib import Path
path = Path(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.append("../")
import deepbayesHF
import deepbayesHF.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow import keras
# Number of data points we want to use:
n_data = 5000
# Range of X values
X_train = np.random.uniform(-4,4,(n_data))
#X_train = np.asarray([X_train])
# Range of noise
noise = np.random.normal(0.0,5,(n_data))
# Computed y values
y_train = [X_train[i]**3 + noise[i] for i in(range(len(X_train)))]
#y_train = np.asarray([y_train])
#X_train = [[i] for i in X_train]
#X_train = np.asarray([X_train])

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

#X_train = tf.convert_to_tensor(X_train)
#y_train = tf.convert_to_tensor(y_train)

learning_rate = 0.0025; decay=0.0; inf=10
opt = optimizers.HamiltonianMonteCarlo()
rob = 0


# A small architecture means fast verification :-)
in_dims = 1
model = Sequential()
#model.add(Input(shape=(None, 1)))
model.add(Dense(50, activation="sigmoid", input_shape=(None,1)))
model.add(Dense(1, activation="linear"))

loss = tf.keras.losses.MeanSquaredError()
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=loss,
)


history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=200,
)
"""

bayes_model = opt.compile(model, loss_fn=loss, epochs=35, learning_rate=learning_rate,
                          mode='regression', burn_in=300, steps=25, b_steps=20)

bayes_model.train(X_train, y_train, X_train, y_train)

bayes_model.save("Posteriors/HMC_Post")
"""
model.save("Posteriors/model4")
