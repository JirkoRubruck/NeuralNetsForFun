import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os


# import requests
# requests.packages.urllib3.disable_warnings()
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_context

# load the data
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()


print(X_train_full.shape,y_train_full.shape)
print(X_test.shape)


def build_model(n_hidden=20, n_neurons=100, learning_rate=0.01):
    """building a model to experiment with the cifar10 dataset"""

    net = keras.models.Sequential()
    net.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for layer in range(n_hidden):
        net.add(keras.layers.BatchNormalization())
        net.add(keras.layers.Dense(n_neurons, activation="selu", kernel_initializer="lecun_normal"))
    net.add(keras.layers.Dense(10, activation="softmax"))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    net.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return net


model = build_model()

# add callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)


def get_run_logdir():
    """
    function from:
    Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow,
    2nd Edition by Aurélien Géron
    """
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


root_logdir = os.path.join(os.curdir, "Cifar_Net_logs")
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)  # callback for tensorboard

history = model.fit(X_train_full, y_train_full, epochs=30, validation_split=0.2,
                    callbacks=[early_stopping_cb, tensorboard_cb])

model.evaluate(X_test, y_test)











