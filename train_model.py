import argparse
import json
import os
import pprint
import time

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

#xtrain = xtrain / 255.0
#xtest = xtest / 255.0
split = -(len(xtrain) // 10)
xtrain, xval = xtrain[:split], xtrain[split:]
ytrain, yval = ytrain[:split], ytrain[split:]

img_shape = xtrain[0].shape
print("img shape", img_shape)
print("img range", xtrain.min(), xtrain.max())
print(len(xtrain), "training images,", len(xval), "validation,", len(xtest), "test")

def train_gen():
    bs = config.batchsize
    i = 0
    while True:
        x = xtrain[i:i+bs]
        y = ytrain[i:i+bs]
        if len(x) < bs:
            i = 0
            x = np.concatenate(x, xtrain[0:bs-len(x)])
            y = np.concatenate(y, ytrain[0:bs-len(x)])
        i += bs
        yield x, y


"""
models
"""

import os
import warnings


def my_xception(input_tensor):
    """Instantiates the Xception architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Note that the default input image size for this model is 299x299.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    backend = K

    input_shape = input_tensor.shape

    channel_axis =  -1

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(input_tensor)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    return x


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)

def mobilenetv2(inpt):
    """
    keras mobilenetv2
    """
    base = keras.applications.MobileNetV2(include_top=False, weights=None,
                input_shape=inpt.shape[1:], alpha=1.0)
    # x = keras.applications.mobilenet_v2.preprocess_input(inpt)
    x = base(inpt)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

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
        x = my_xception(inpt)
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
            keras.metrics.BinaryAccuracy(),
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
            x=train_gen(),
            validation_data=(xval, yval),
            batch_size=config.batchsize,
            epochs=config.epochs,
            verbose=1,
            steps_per_epoch=1000,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("\nManual early stopping")
        H = callbacks[0]
    end = time.time()
    
    # save training stats
    os.makedirs("stats/"+args.name, exist_ok=True)
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


