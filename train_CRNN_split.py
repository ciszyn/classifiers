import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from variables import genres, spec_path, set_size
from joblib import dump
from ConsecutiveEarlyStopping import ConsecutiveEarlyStopping
import keras.backend as K


def train():
    model_path = './models/'+set_size+'/fma.crnn_split'
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory(spec_path + 'training/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    val_it = datagen.flow_from_directory(spec_path + 'validation/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    test_it = datagen.flow_from_directory(spec_path + 'test/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))

    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print('no model found')

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
            y.append(layers.Bidirectional(layers.GRU(64, return_sequences=True))(x[:,i,:,:]))
            y[i] = layers.Bidirectional(layers.GRU(64))(y[i])
            y[i] = layers.Flatten()(y[i])

        z = layers.concatenate(y)
        z = layers.Dense(64, activation='relu')(z)
        z = layers.Dense(64, activation='relu')(z)
        z = layers.Dense(len(genres), activation='softmax')(z)

        model = tf.keras.Model(inputs=[input], outputs=[z])


    callback = ConsecutiveEarlyStopping(monitor='val_accuracy', patience=3, mode='max', model_path=model_path)

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_it, epochs=20, validation_data=val_it, callbacks=[])
    try:
        dump(history.history, "./history/" + set_size + '/' + 'crnn_split.joblib')
    except:
        os.makedirs("./history/" + set_size)
        dump(history.history, "./history/" + set_size + '/' + 'crnn_split.joblib')
    model.save('./models/'+set_size+'/fma.crnn_split')
    

if __name__ == "__main__":
    train()