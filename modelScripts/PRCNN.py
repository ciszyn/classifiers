from keras.models import Sequential
from keras import layers
import tensorflow as tf


def getModel(output_size):
    input = tf.keras.Input(shape=(128, 259, 1))

    x = layers.Conv2D(16, (3, 1), padding='same')(input)
    x = layers.MaxPooling2D((2, 2), (2, 2))(x)
    x = layers.Conv2D(32, (3, 1), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), (2, 2))(x)
    x = layers.Conv2D(64, (3, 1), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), (2, 2))(x)
    x = layers.Conv2D(128, (3, 1), padding='same')(x)
    x = layers.MaxPooling2D((4, 4), (4, 4))(x)
    x = layers.Conv2D(64, (3, 1), padding='same')(x)
    x = layers.MaxPooling2D((4, 4), (4, 4))(x)
    x = layers.Flatten()(x)

    y = layers.MaxPooling2D((1, 2), (1, 2))(input)
    y = layers.Reshape((128, 129))(y)
    y = layers.Bidirectional(layers.GRU(64))(y)

    z = layers.concatenate([x, y])
    z = layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.Model(inputs=[input], outputs=[z])
    return model
