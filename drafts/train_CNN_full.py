import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from variables import *
from joblib import dump
from ConsecutiveEarlyStopping import ConsecutiveEarlyStopping


def train():
    model_path = './models/'+set_size+'/fma.cnn_full'
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory(spec_full_path + 'training/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 2582))
    val_it = datagen.flow_from_directory(spec_full_path + 'validation/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 2582))
    test_it = datagen.flow_from_directory(spec_full_path + 'test/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 2582))

    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 2582, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(genres), activation='softmax'))

    try:
        model.load_model(model_path)
    except:
        print('no saved model')

    callback = ConsecutiveEarlyStopping(monitor='val_accuracy', patience=3, mode='max', model_path=model_path)

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_it, epochs=20, validation_data=val_it, callbacks=[callback])
    try:
        dump(history.history, "./history/" + set_size + '/' + 'cnn_full.joblib')
    except:
        os.makedirs("./history/" + set_size)
        dump(history.history, "./history/" + set_size + '/' + 'cnn_full.joblib')

    model.save('./models/'+set_size+'/fma.cnn_full')
    

if __name__ == "__main__":
    train()