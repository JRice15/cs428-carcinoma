import argparse
import json
import os
import time
import pprint

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (History, LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, EarlyStopping)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from cv_helpers import *
from models import make_model
from save_stats import save_history

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--load",action="store_true")
parser.add_argument("--nodisplay",action="store_true")
parser.add_argument("--test",action="store_true",help="load a small portion of the data for a quick test run")
args = parser.parse_args()

class TrainConfig:

    def __init__(self, epochs, model, batchsize, lr, lr_sched_freq, 
            lr_sched_factor, loss):
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

x = []
y = []
for name in os.listdir("data"):
    for label in os.listdir("data/"+name):
        ylabel = int(label)
        for imname in os.listdir("data/"+name+"/"+label):
            path = "data/"+name+"/"+label+"/"+imname
            im = cv.imread(path)
            x.append(im)
            y.append(ylabel)

x = np.array(x)
y = np.array(y)

x, xtest, y, ytest = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=12)
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.15, random_state=33)

# free memory
del x, y


img_shape = xtrain[0].shape
print("img_shape", img_shape)
print(len(xtrain), "training images,", len(xval), "validation,", len(xtest), "test")

"""
make model
"""


if not args.load:

    model = make_model(config.model, img_shape)

    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(config.lr),
        metrics=[
            keras.metrics.TruePositives(),
            keras.metrics.TrueNegatives(),
            keras.metrics.FalsePositives(),
            keras.metrics.FalseNegatives(),            
        ],
    )


    """
    train model
    """

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
    
    step = max(1, len(H.history['loss']) // 6)
    save_history(H, args.name, end-start, config, marker_step=step)



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


