from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy,os

def ANNModelParams(x, y):
    """
    The Tensor flow for model check
    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    ####

    x_train, x_test = x
    y_train, y_test = y
    print(x_train.shape)
    print(y_test.shape)

    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(x_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss='logcosh', optimizer='Nadam', metrics=['accuracy'])
    # adding an early stop so that it does not overfit
    ES = EarlyStopping(monitor='val_loss', patience=100)
    # checkpoint
    CP = ModelCheckpoint('Checkpoint.h5', verbose=0, save_best_only=True)
    # Reduce learning rate
    RL = ReduceLROnPlateau(factor=0.2)

    model.fit(x_train, y_train, epochs=int(2e6), verbose=0, shuffle="batch", callbacks=[ES, CP, RL],
              validation_data=(numpy.array(x_test), numpy.array(y_test)))

    return model
