import util
import matplotlib.pyplot as plt
import numpy
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import seaborn as sn
from variables import *
from os import listdir
from measure import measure_perf

def evaluate(model_type):
    datagen = ImageDataGenerator(rescale=1./255)
    test_it = datagen.flow_from_directory(spec_path + 'test/', color_mode='grayscale', class_mode='categorical', batch_size=32, target_size=(128, 259))


    model = models.load_model('./models/'+set_size+'/'+'fma.'+model_type)
    print('loaded')
    result = {}
    labels = genres
    for genre in labels:
        if genre == 'Old-Time / Historic':
            genre = 'Old-Time - Historic'
        for file in listdir(spec_path + 'test/' + genre + '/'):
            file = spec_path + 'test/' + genre + '/' + file
            img = load_img(file, color_mode='grayscale', target_size=(128, 259))
            input_arr = img_to_array(img)
            input_arr /= 255.
            input_arr = numpy.array([input_arr])
            id = file.split('_')[2]
            if id not in result:
                result[id] = [[0]*len(labels), 0]
                result[id][1] = labels.index(tracks['track', 'genre_top'].loc[int(id)])
            prediction = model.predict(input_arr)
            result[id][0][argmax(prediction)] += 1

    matrix = [[0]*len(labels) for i in range(len(labels))]

    for track in result:
        predicted = argmax(result[track][0])
        real = result[track][1]
        matrix[real][predicted] += 1
            
    scores = {}
    name = model_type
    measures = ["accuracy", "TP_rate", "FP_rate", "Precision", "F_measure"]
    for measure in measures:
        scores[measure] = pd.read_csv('./results/' + set_size + '/' + measure + '.csv', index_col=0)

    measure_perf(name, scores, matrix)
    for measure in measures:
        scores[measure].to_csv('./results/' + set_size + '/' + measure + '.csv')

    df_cm = pd.DataFrame(matrix, index = genres,
                    columns = genres)
    plt.figure(figsize = (10,7))
    plt.xticks(rotation=45)
    ax = sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    ax.figure.tight_layout()
    plt.savefig('./matrices/'+ set_size + '/' + name + '.png')


if __name__ == "__main__":
    evaluate('crnn9')