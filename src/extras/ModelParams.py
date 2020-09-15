
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *



def ANNModelParams(x, y):
    """
    The Tensor flow for parameter estimation

    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    ####
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
    model.add(Dense(y.shape[1]))
    model.compile(loss='logcosh', optimizer='Nadam', metrics=['accuracy'])
    # adding an early stop so that it does not over fit. slow but better
    # ES = EarlyStopping(monitor='val_loss', patience=100)
    # model.fit(x, y, epochs=int(2e6), verbose=2, shuffle="batch", callbacks=[ES], validation_split=.1)
    ####
    model.fit(x, y, epochs=100, shuffle=True,  verbose=2)

    return model
