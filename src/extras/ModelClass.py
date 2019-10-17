from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *

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
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=sd, dtype=tf.float32)
    return input_layer + noise


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
    # adding an early stop so that it does not over fit. slow but better
    # ES = EarlyStopping(monitor='val_loss', patience=100)
    # model.fit(x, y, epochs=int(2e6), verbose=2, shuffle="batch", callbacks=[ES], validation_split=.1)
    ####
    model.fit(x, y, epochs=5, verbose=2, shuffle="batch")

    return model
