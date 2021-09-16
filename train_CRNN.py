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
    model_path = './models/'+set_size+'/fma.crnn'
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory(spec_path + 'training/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    val_it = datagen.flow_from_directory(spec_path + 'validation/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    test_it = datagen.flow_from_directory(spec_path + 'test/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))

    model = Sequential()
    model.add(layers.Conv2D(64, (3,3), activation='elu', input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='elu', input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='elu', input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='elu', input_shape=(128, 259, 1)))
    model.add(layers.Dropout(.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Reshape((84, 128)))
    model.add(layers.GRU(64, return_sequences=True))
    model.add(layers.Dropout(.3))
    model.add(layers.GRU(64, return_sequences=True))
    model.add(layers.Dropout(.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(8, activation='softmax'))


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
        dump(history.history, "./history/" + set_size + '/' + 'crnn.joblib')
    except:
        os.makedirs("./history/" + set_size)
        dump(history.history, "./history/" + set_size + '/' + 'crnn.joblib')
    model.save('./models/'+set_size+'/fma.crnn')
    

if __name__ == "__main__":
    train()