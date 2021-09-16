import matplotlib.pyplot as plt
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import seaborn as sn
from variables import *
from measure import measure_perf

def evaluate():
    datagen = ImageDataGenerator(rescale=1./255)
    test_it = datagen.flow_from_directory(spec_full_path + 'test/', color_mode='grayscale', class_mode='categorical', batch_size=1, target_size=(128, 2582))

    model = models.load_model('./models/'+set_size+'/'+'fma_full.cnn')
    labels = genres

    matrix = [[0]*len(labels) for i in range(len(labels))]
    count = found = 0

    print(test_it.samples)
    i = 0

    for x, y in test_it:
        predicted = argmax(model.predict(x))
        real = argmax(y)
        matrix[real][predicted] += 1
        if i >= test_it.samples:
            break
        else:
            i+=1
            
    scores = {}
    name = 'cnn_full'
    measures = ["accuracy", "TP_rate", "FP_rate", "Precision", "F_measure", 'Kappa']
    for measure in measures:
        scores[measure] = pd.read_csv('./results/' + set_size + '/' + measure + '.csv', index_col=0)

    measure_perf(name, scores, matrix)
    for measure in measures:
        scores[measure].to_csv('./results/' +  set_size + '/' + measure + '.csv')

    df_cm = pd.DataFrame(matrix, index = genres,
                    columns = genres)
    plt.figure(figsize = (10,7))
    plt.xticks(rotation=45)
    ax = sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    ax.figure.tight_layout()
    plt.savefig('./matrices/' + set_size + '/' + name+ '.png')

if __name__ == "__main__":
    evaluate()