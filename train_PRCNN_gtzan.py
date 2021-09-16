import tensorflow as tf
from keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
from ConsecutiveEarlyStopping import ConsecutiveEarlyStopping
from variables import *
from joblib import dump

def train():
    model_path = './models/'+set_size+'/gtzan.prcnn'
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory('/media/misiek/Dane/gtzan_spec/train/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))
    val_it = datagen.flow_from_directory('/media/misiek/Dane/gtzan_spec/test/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))

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
    y = layers.Bidirectional(layers.GRU(64))(y)        #well this may be a lot different

    z = layers.concatenate([x, y])
    z = layers.Dense(10, activation='softmax')(z)

    model = tf.keras.Model(inputs=[input], outputs=[z])


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
        dump(history.history, "./history/" + set_size + '/' + 'prcnn_gtzan.joblib')
    except:
        os.makedirs("./history/" + set_size)
        dump(history.history, "./history/" + set_size + '/' + 'prcnn_gtzan.joblib')
    model.save('./models/'+set_size+'/gtzan.prcnn')
    

if __name__ == "__main__":
    train()