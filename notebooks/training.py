import os
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def test_train_split(X, y, test_size=0.2, random_state=42):
    """Split a dataset into train and test"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=10, learning_rate=1e-3, output_dir=None):
    """Train a model"""
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    callbacks = []
    if output_dir is not None:
        # remove existing models
        os.makedirs(output_dir, exist_ok=True)
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(output_dir, 'best.h5'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ))
    callbacks.append(
        EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ))
    model.fit(X_train.reshape(-1, 28, 28, 1), to_categorical(y_train), batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(X_test.reshape(-1, 28, 28, 1), to_categorical(y_test)),
              callbacks=callbacks)
    return model