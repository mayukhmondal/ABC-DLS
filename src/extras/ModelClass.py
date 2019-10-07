from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.keras.callbacks import Callback


def Gaussian_noise(input_layer, sd: float = .01):
    """
    Gaussian noise to the input data. Same as Keras.GaussianNoise but it will not only work with training part but
    will work on test data set and observed data. Thus every time it will run will give slightly different results.
    Good to produce a distribution from a single observation
    :param input_layer: tensorflow input layer
    :param sd: the standard deviation present will be present in the noise random normal distribution
    :return: will add the noise to the input_layer
    """
    import tensorflow as tf
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=sd, dtype=tf.float32)
    return input_layer + noise

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
    model.add(Lambda(Gaussian_noise, input_shape=(x.shape[1],)))
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
    model.fit(x, y, epochs=50, verbose=2, shuffle="batch")

    return model
