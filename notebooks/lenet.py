from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def lenet_5(dropout_rate=0.5):    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(120, activation='relu'))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

