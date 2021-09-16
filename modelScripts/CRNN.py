from keras.models import Sequential
from keras import layers


def getModel(output_size):
    model = Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='elu',
              input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='elu',
              input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='elu',
              input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='elu',
              input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Reshape((84, 128)))
    model.add(layers.GRU(64, return_sequences=True))
    model.add(layers.Dropout(.3))
    model.add(layers.GRU(64, return_sequences=True))
    model.add(layers.Dropout(.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_size, activation='softmax'))
    return model
