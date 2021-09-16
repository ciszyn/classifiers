from modelScripts.PRCNN import getModel
from joblib import dump
from variables import *
from ConsecutiveEarlyStopping import ConsecutiveEarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train():
    model_path = './models/'+set_size+'/fma.prcnn'
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory(
        spec_path + 'training/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    val_it = datagen.flow_from_directory(
        spec_path + 'validation/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))

    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print('no saved model')
        model = getModel(len(genres))

    callback = ConsecutiveEarlyStopping(
        monitor='val_accuracy', patience=3, mode='max', model_path=model_path)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_it, epochs=20,
                        validation_data=val_it, callbacks=[callback])
    try:
        dump(history.history, "./history/" + set_size + '/' + 'prcnn.joblib')
    except:
        os.makedirs("./history/" + set_size)
        dump(history.history, "./history/" + set_size + '/' + 'prcnn.joblib')
    model.save('./models/'+set_size+'/fma.prcnn')


if __name__ == "__main__":
    train()
