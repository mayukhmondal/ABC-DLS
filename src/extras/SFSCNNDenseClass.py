
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau



def ANNModelCheck(x, y):
    """
    The Tensor flow for model check
    :param x: the x or summary statistics
    :param y: the y or model names or classification
    :return: will return the trained model
    """
    x_0 = Input(shape=(x.shape[1],))    
    # Dense
    den=GaussianNoise(0.05)(x_0)
    den=ReLU()(den)
    den = Dense(128, activation='relu')(den)
    den = Dense(64, activation='relu')(den)
    den = Dense(32, activation='relu')(den)
    den = Dense(16, activation='relu')(den)
    den=Model(inputs=x_0,outputs=den)
    # CNN
    sfsdim=(11,11,11,1)
    cnn=GaussianNoise(0.05)(x_0)
    cnn=ReLU()(cnn)
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
    combined=Dense(y.shape[1], activation='softmax')(combined)
    model=Model(inputs=x_0,outputs=combined)
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['categorical_accuracy'])
    # adding an early stop so that it does not overfit
    ES = EarlyStopping(monitor='val_loss', patience=20)
    # checkpoint
    CP = ModelCheckpoint('Checkpoint.h5', verbose=1, save_best_only=True)
    # Reduce learning rate
    RL = ReduceLROnPlateau(factor=0.2,patience=5)

    model.fit(x, y, epochs=int(2e6), verbose=2, shuffle=True, callbacks=[ES, CP, RL], validation_split=.2)
    return model
