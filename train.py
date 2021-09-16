from ConsecutiveEarlyStopping import ConsecutiveEarlyStopping
from joblib import dump
from variables import *
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from modelScripts.CNN import getModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(model_name, getModel):
    model_path = './models/'+set_size+'/fma.'+model_name
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory(
        spec_path + 'training/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    val_it = datagen.flow_from_directory(
        spec_path + 'validation/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))

    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print('no model found')
        model = getModel(len(genres))

    callback = ConsecutiveEarlyStopping(
        monitor='val_accuracy', patience=3, mode='max', model_path=model_path)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_it, epochs=6,
                        validation_data=val_it, callbacks=[callback])

    try:
        dump(history.history, "./history/" +
             set_size + '/' + model_name + '.joblib')
    except:
        os.makedirs("./history/" + set_size)
        dump(history.history, "./history/" +
             set_size + '/' + model_name + '.joblib')
    model.save('./models/'+set_size+'/fma.'+model_name)


if __name__ == "__main__":
    train('cnn', getModel)
