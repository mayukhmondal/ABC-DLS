
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True



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


def ANNModelCheck(x, y):
    """
    The Tensor flow for model check
    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    
    x_0 = Input(shape=(x.shape[1],))
    # Dense
    den=Lambda(Gaussian_noise)(x_0)
    den = Dense(128, activation='relu')(den)
    den = Dense(64, activation='relu')(den)
    den = Dense(32, activation='relu')(den)
    den = Dense(16, activation='relu')(den)
    den=Model(inputs=x_0,outputs=den)
    # CNN
    sfsdim=(11,11,11,1)
    cnn=Lambda(Gaussian_noise)(x_0)
    cnn=Reshape(sfsdim)(cnn)
    cnn=Conv3D(14, (3, 3, 3), activation='relu', padding='same')(cnn)
    cnn=MaxPooling3D((3, 3, 3), padding='same')(cnn)
    cnn=Flatten()(cnn)
    cnn=Dense(32, activation='relu')(cnn)
    cnn=Dense(16, activation='relu')(cnn)
    cnn=Model(inputs=x_0,outputs=cnn)
    
    # togehter
    combined=Concatenate()([den.output,cnn.output])
    combined=Dense(16, activation='relu')(combined)
    combined=Dense(8, activation='relu')(combined)
    combined=Dense(4, activation='relu')(combined)
    combined=Dense(y.shape[1], activation='softmax')(combined)
    model=Model(inputs=x_0,outputs=combined)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    ###adding an early stop so that it does not overfit
    ES = EarlyStopping(monitor='val_loss', patience=100)
    ####
    model.fit(x, y, epochs=int(2e6), verbose=2, shuffle="batch", callbacks=[ES], validation_split=.2)

    return model
