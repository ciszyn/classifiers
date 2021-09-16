from keras.models import Sequential
from keras import layers


def getModel(output_size):
    model = Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu',
              input_shape=(128, 259, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Reshape((105, 32)))
    model.add(layers.Bidirectional(layers.GRU(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.GRU(64)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_size, activation='softmax'))
    return model
