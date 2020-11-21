import argparse
import json
import os
import pprint
import time

import cv2
cv = cv2
import logging

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.applications import imagenet_utils
from keras.callbacks import (EarlyStopping, History, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, Lambda, MaxPooling2D,
                          Multiply, ReLU, Reshape, Softmax)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

if not tf.__version__.startswith("2.2") or not keras.__version__.startswith("2.4.3"):
    print("This code was written with TensorFlow 2.2 and Keras 2.4.3, and may fail on your version:")
print("tf:", tf.__version__)
print("keras:", keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--load",action="store_true")
parser.add_argument("--nodisplay",action="store_true")
parser.add_argument("--test",action="store_true",help="load a small portion of the data for a quick test run")
args = parser.parse_args()

class TrainConfig:

    def __init__(self, epochs, model, batchsize, lr, lr_sched_freq, 
            lr_sched_factor):
        self.epochs = epochs
        self.model = model
        self.batchsize = batchsize
        self.lr = lr
        self.lr_sched_freq = lr_sched_freq
        self.lr_sched_factor = lr_sched_factor
        pprint.pprint(vars(self))
    
    def __str__(self):
        return str(vars(self))
    
    def write_to_file(self,filename):
        with open(filename, "a") as f:
            f.write("\n" + str(self) + "\n\n")

with open("model_config.json", "r") as f:
    config_dict = json.load(f)

config = TrainConfig(**config_dict)

"""
load data
"""

from keras.utils import get_file

x_train_path = get_file('idc_train.h5','https://storage.googleapis.com/cpe428-fall2020-datasets/idc_train.h5')
x_test_path = get_file('idc_test.h5','https://storage.googleapis.com/cpe428-fall2020-datasets/idc_test.h5')

import h5py as h5

with h5.File(x_train_path,'r') as f:
  xtrain = f['X'][:,1:49,1:49]
  ytrain = f['y'][:]
with h5.File(x_test_path,'r') as f:
  xtest = f['X'][:,1:49,1:49]
  ytest = f['y'][:]

xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, validation_split=0.10, shuffle=True)

img_shape = xtrain[0].shape
print("img_shape", img_shape)
print(len(xtrain), "training images,", len(xval), "validation,", len(xtest), "test")

"""
models
"""


def xception(inpt):
    """
    keras xception network. see https://keras.io/api/applications/
    """
    base = keras.applications.Xception(include_top=False, weights=None, 
                input_shape=inpt.shape[1:])
    x = base(inpt)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    return x

def mobilenetv2(inpt):
    """
    keras mobilenetv2
    """
    base = keras.applications.MobileNetV2(include_top=False, weights=None,
                input_shape=inpt.shape[1:], alpha=1.0)
    # x = keras.applications.mobilenet_v2.preprocess_input(inpt)
    x = base(inpt)
    x = GlobalAveragePooling2D()(x)

    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    return x



def make_model(name, input_shape):
    """
    get model from case insensitive name
    args:
        model name
        input shape (not including batch size)
        output_confidences (bool): whether loss to output a prediction, or a 
            softmax confidence for each string
    """
    inpt = Input(input_shape)

    name = name.lower()
    if name == "xception":
        x = xception(inpt)
    elif name == "mobilenetv2":
        x = mobilenetv2(inpt)
    else:
        raise ValueError("no model named '" + name + "'")

    # all models output a vector of size 256
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(16)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(1)(x)
    x = ReLU()(x)
    x = Activation('sigmoid')(x)

    return Model(inpt, x)


"""
run training
"""

if not args.load:

    model = make_model(config.model, img_shape)

    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(config.lr),
        metrics=[
            keras.metrics.Accuracy(),
            keras.metrics.TruePositives(),
            keras.metrics.TrueNegatives(),
            keras.metrics.FalsePositives(),
            keras.metrics.FalseNegatives(),            
        ],
    )

    def lr_sched(epoch, lr):
        if epoch == 0:
            pass
        elif epoch % config.lr_sched_freq == 0:
            lr = lr * config.lr_sched_factor
            print("Decreasing learning rate to", lr)
        return lr

    os.makedirs("models/", exist_ok=True)
    callbacks = [
        History(),
        LearningRateScheduler(lr_sched),
        ModelCheckpoint("models/"+args.name+".hdf5", save_best_only=True, verbose=1, period=1),
        EarlyStopping(monitor='val_loss', verbose=1, patience=int(config.lr_sched_freq * 1.5))
    ]

    start = time.time()
    try:
        H = model.fit(
            xtrain,
            ytrain,
            validation_data=(xval, yval),
            batch_size=config.batchsize,
            epochs=config.epochs,
            verbose=1,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("\nManual early stopping")
        H = callbacks[0]
    end = time.time()
    
    # save training stats
    os.makedirs("stats/", exist_ok=True)
    plt.plot(H.history["loss"])
    plt.plot(H.history["val_loss"])
    plt.legend(['train', 'val'])
    plt.title("Loss")
    plt.savefig("stats/"+args.name+"/loss.png")

    plt.plot(H.history["accuracy"])
    plt.plot(H.history["val_accuracy"])
    plt.legend(['train', 'val'])
    plt.title("Accuracy")
    plt.savefig("stats/"+args.name+"/accuracy.png")

    statsfile_name = "stats/" + args.name + "/stats.txt"
    secs = end - start
    epochs_ran = len(H.history["loss"])
    with open(statsfile_name, "w") as f:
        f.write(args.name + "\n\n")
        f.write("Epochs ran: {}\n".format(epochs_ran))
        f.write("Secs per epoch: {}\n".format(secs / epochs_ran))
        f.write("Minutes total: {}\n".format(secs / 60))
        f.write("Hours total: {}\n".format(secs / 3600))
    config.write_to_file(statsfile_name)


print("Loading model...")
model = keras.models.load_model("models/"+args.name+".hdf5")

if args.load:
    # if we are just loading and have not trained
    model.summary()


"""
testing
"""

print("Evaluating on test set")
results = model.evaluate(xtest, ytest)
with open("stats/"+args.name+"/stats.txt", "a") as f:
    f.write("\nTest results:\n")
    for i,name in enumerate(model.metrics_names):
        print(" ", name+":", results[i])
        f.write(name+": "+str(results[i])+"\n")


