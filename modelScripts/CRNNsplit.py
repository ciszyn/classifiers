from keras.models import Sequential
from keras import layers
import tensorflow as tf


def getModel(output_size):
    input = tf.keras.Input(shape=(128, 259, 1))
    x = layers.Conv2D(32, (5, 5), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (5, 5), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    y = []

    for i in range(7):
        y.append(layers.Bidirectional(layers.GRU(
            64, return_sequences=True))(x[:, i, :, :]))
        y[i] = layers.Bidirectional(layers.GRU(64))(y[i])
        y[i] = layers.Flatten()(y[i])

    z = layers.concatenate(y)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.Model(inputs=[input], outputs=[z])
    return model
