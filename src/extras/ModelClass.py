import sklearn
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D, Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import tensorflow
import scipy
from sklearn import preprocessing
from tensorflow.keras.utils import HDF5Matrix


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True





def ANNModelCheck(x, y):
    """
    The Tensor flow for model check
    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(x.shape[1],)))
    model.add(Dropout(.01))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs=20, verbose=2, shuffle="batch")

    return model
