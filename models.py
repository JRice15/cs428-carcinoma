import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, Lambda, MaxPooling2D,
                          Multiply, ReLU, Reshape, Softmax)
from keras.models import Model
import logging
from keras.optimizers import Adam
from keras.applications import imagenet_utils

if not tf.__version__.startswith("2.2") or not keras.__version__.startswith("2.4.3"):
    print("This code was written with TensorFlow 2.2 and Keras 2.4.3, and may fail on your version:")
print("tf:", tf.__version__)
print("keras:", keras.__version__)


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




