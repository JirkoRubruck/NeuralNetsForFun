import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
build a CNN to train on the CIFAR10 dataset to compare performance with 
a dense deep Network
"""
# load the data
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

print("train:", X_train_full.shape, y_train_full.shape)
print("test:", X_test.shape, y_test.shape)

#  some exploration: show some images from the dataset
for im in range(6):
    plt.subplot(231+im)
    plt.imshow(X_train_full[im])
plt.show()

"""building a convolutional neural net in inspired by AlexNet"""

model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(96, 8, strides=2, activation="elu", padding="same",
                        input_shape=[32,32,3]),
    keras.layers.MaxPool2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, 5, activation="elu", padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(384, 3, activation="elu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(384, 3, activation="elu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, 3, activation="elu", padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation="elu"),
    keras.layers.Dropout(0.5),          # regularisation
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation="elu"),
    keras.layers.Dropout(0.5),                      # regularisation
    keras.layers.Dense(10, activation="softmax")    # softmax for output
])

optimizer = keras.optimizers.Adam(lr=0.05)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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

history = model.fit(X_train_full, y_train_full, batch_size=15, epochs=30, validation_split=0.2,
                    callbacks=[early_stopping_cb, tensorboard_cb])

model.evaluate(X_test, y_test)


