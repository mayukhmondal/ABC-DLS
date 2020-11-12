from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def ANNModelCheck(x, y):
    """
    The Tensor flow for model check

    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    print(x.shape)
    model = Sequential()
    model.add(GaussianNoise(.01, input_shape=(x.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.01))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
    # adding an early stop so that it does not over fit. slow but better
    # ES = EarlyStopping(monitor='val_loss', patience=100)
    # model.fit(x, y, epochs=int(2e6), verbose=2,callbacks=[ES], validation_split=.1)
    ####
    model.fit(x, y, epochs=10, verbose=2, shuffle=True)

    return model
