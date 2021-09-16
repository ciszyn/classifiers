import matplotlib.pyplot as plt
from joblib import load
from variables import *
import random

def plot(model_name):
    history = load("./history/"+set_size+'/'+model_name+'.joblib')
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(model_name)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./history/"+set_size+'/'+model_name+'.png')
    plt.close()

if __name__ == "__main__":
    plot("crnn9")