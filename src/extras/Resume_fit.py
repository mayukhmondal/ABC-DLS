import numpy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *


def model_fit(save_model, x, y):
    """
    The Tensor flow for parameter estimation

    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    ####
    x_train, x_test = x
    y_train, y_test = y
    # adding an early stop so that it does not overfit
    ES = EarlyStopping(monitor='val_loss', patience=100)
    # checkpoint
    CP = ModelCheckpoint('Checkpoint.h5', verbose=0, save_best_only=True)
    # Reduce learning rate
    RL = ReduceLROnPlateau(factor=0.2)
    save_model.fit(x_train, y_train, epochs=10, verbose=1, shuffle=True, callbacks=[ES, CP, RL],
                   validation_data=(numpy.array(x_test), numpy.array(y_test)))

    return save_model
