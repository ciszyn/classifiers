import numpy
from keras.preprocessing.image import load_img, img_to_array
from pandas import DataFrame
from variables import *
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def extract():
    model_path = './models/'+set_size+'/fma.crnn4'
    model = tf.keras.models.load_model(model_path)
    model = tf.keras.models.Sequential(model.layers[:-1])
    model.summary()
    model.layers[-1].activation = tf.keras.activations.sigmoid
    model.save("tmp")
    model = tf.keras.models.load_model("tmp")
    model.summary()
    x = {}
    y = {}
    for split in os.listdir(spec_path):
        print(split)
        if split not in x:
            x[split] = DataFrame()
            y[split] = DataFrame()
        for genre in os.listdir(spec_path+split+'/'):
            for spectrogram in os.listdir(spec_path+split+'/'+genre+'/'):
                file = spec_path+split+'/'+genre+'/'+spectrogram
                img = load_img(file, color_mode='grayscale',
                               target_size=(128, 259))
                input_arr = img_to_array(img)
                input_arr /= 255.
                input_arr = numpy.array([input_arr])
                features = model.predict(input_arr)
                x[split] = x[split].append(
                    DataFrame(features), ignore_index=True)
                y[split] = y[split].append([genre], ignore_index=True)
    for split in x:
        x[split].to_csv('./features/x_'+split+'.csv')
        y[split].to_csv('./features/y_'+split+'.csv')


if __name__ == '__main__':
    extract()
