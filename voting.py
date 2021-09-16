#import tensorflow as tf
from keras import models
from variables import *
from os import listdir
from keras.preprocessing.image import load_img, img_to_array
import numpy
from numpy import argmax
from joblib import load, dump
from numpy import tensordot
from sklearn.metrics import accuracy_score
from numpy.linalg import norm
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import minimize
import numpy as np
from joblib import dump
from extractVoting import extractVoting, votingSummary
from votingConfig import optimizationMethods


def plurarity(x):
    ones = []
    for model in x:
        model_res = []
        for sample in model:
            result = np.zeros_like(sample)
            result[np.argmax(sample)] = 1
            sample = result
            model_res.append(sample)
        ones.append(model_res)
    return np.array(ones)


def kApproval(x):
    ones = []
    for model in x:
        model_res = []
        for sample in model:
            result = np.zeros_like(sample)
            result[sample.argsort()[-1]] = 3
            result[sample.argsort()[-2]] = 2
            result[sample.argsort()[-3]] = 1
            sample = result
            model_res.append(sample)
        ones.append(model_res)
    return np.array(ones)


def normalize(weights):
    result = norm(weights, 1)
    if result == 0.0:
        return weights
    return weights / result


def keras_predict(model, split):
    result = {}
    labels = genres
    for genre in labels:
        if genre == 'Old-Time / Historic':
            genre = 'Old-Time - Historic'
        for file in listdir(spec_path + split + '/' + genre + '/'):
            file = spec_path + split + '/' + genre + '/' + file
            img = load_img(file, color_mode='grayscale',
                           target_size=(128, 259))
            input_arr = img_to_array(img)
            input_arr /= 255.
            input_arr = numpy.array([input_arr])
            id = file.split('_')[2]
            if id not in result:
                result[id] = [0]*len(labels)
            prediction = model.predict(input_arr)
            result[id][argmax(prediction)] += 1
    output = {}
    for id in result:
        output[int(id)] = result[id]

    return output


def sklearn_predict(model, split):
    res = {}
    for id in split:
        val = model.predict(
            [features.loc[id, list(features.columns.levels[0])].to_numpy()])[0]
        res[id] = [0]*len(genres)
        res[id][val] = 1
    return res

# Borda


def evaluate_weights_Borda(weights, yhats, y):
    yhats = np.array(yhats)

    n_weights = normalize(weights)
    summed = tensordot(yhats, n_weights, axes=((0), (0)))

    yhat = argmax(summed, axis=1)
    return 1 - accuracy_score(y, yhat)

# Plurarity


def evaluate_weights_Plurarity(weights, yhats, y):
    n_weights = normalize(weights)
    yhats = np.array(yhats)
    yhats = plurarity(yhats)
    summed = tensordot(yhats, n_weights, axes=((0), (0)))

    yhat = argmax(summed, axis=1)
    return 1 - accuracy_score(y, yhat)


def evaluate_weights_kApproval(weights, yhats, y):
    n_weights = normalize(weights)
    yhats = np.array(yhats)
    yhats = kApproval(yhats)
    summed = tensordot(yhats, n_weights, axes=((0), (0)))

    yhat = argmax(summed, axis=1)
    return 1 - accuracy_score(y, yhat)


def prepare_data(keras_models, sklearn_models, split_set, split_name):
    labels = tracks['track', 'genre_top']
    y = {}
    yhats = []

    for id in split_set:
        enc = LabelEncoder().fit(labels)
        y[id] = enc.transform([labels[id]])[0]

    for model in keras_models:
        model = models.load_model('./models/'+set_size+'/'+'fma.'+model)
        yhats.append(keras_predict(model, split_name))
    for model in sklearn_models:
        model = load('./models/'+set_size+'/'+model+'.joblib')
        yhats.append(sklearn_predict(model, split_set))

    outputs = []
    for model_prediction in yhats:
        outputs.append([])
        for id in split_set:
            try:
                outputs[-1].append(model_prediction[id])
            except:
                outputs[-1].append([1/len(genres)]*len(genres))

    real_outputs = []
    for id in split_set:
        real_outputs.append(y[id])

    return outputs, real_outputs


def main(evaluate_weights, method):
    keras_models = ['crnn', 'crnn2', 'crnn3', 'crnn4', 'crnn5', 'prcnn']
    sklearn_models = ['LR', 'SVCpoly1', 'SVCrbf', 'linSVC2', 'MLP1', 'MLP2']
    models = keras_models + sklearn_models

    num_models = len(keras_models) + len(sklearn_models)

    try:
        outputs = load('./votingVars/predictions_val.joblib')
        real_outputs = load('./votingVars/labels_val.joblib')
        outputs_test = load('./votingVars/predictions_test.joblib')
        real_outputs_test = load('./votingVars/labels_test.joblib')
    except:
        outputs, real_outputs = prepare_data(
            keras_models, sklearn_models, val, 'validation')
        dump(outputs, './votingVars/predictions_val.joblib')
        dump(real_outputs, './votingVars/labels_val.joblib')
        outputs_test, real_outputs_test = prepare_data(
            keras_models, sklearn_models, test, 'test')
        dump(outputs_test, './votingVars/predictions_test.joblib')
        dump(real_outputs_test, './votingVars/labels_test.joblib')

    classifier_index = [i for i in range(len(outputs))]

    index_combinations = [[]]
    combinations = [[]]
    for output, index in zip(outputs, classifier_index):
        combinations += [temp + [output] for temp in combinations]
        index_combinations += [temp + [index] for temp in index_combinations]
    result = 1.

    i = 0

    combinations = combinations[1:]
    index_combinations = index_combinations[1:]

    best_weights = []
    best_combinations = []
    performances = []

    for combination, index_combination in zip(combinations, index_combinations):
        if (len(combination) % 2 == 0):
            continue
        weights = [1/len(combination) for _ in range(len(combination))]
        loss = evaluate_weights(weights, combination, real_outputs)
        testLoss = evaluate_weights(
            weights, [outputs_test[i] for i in index_combination], real_outputs_test)
        performances.append((loss, index_combination, combination, testLoss))
    performances.sort(key=lambda tup: tup[0], reverse=False)

    results = {}

    count = 1
    if method == 'differential_evolution':
        count = 5

    for oldLoss, index_combination, combination, oldTestLoss in performances[:50]:
        i += 1
        bound_w = [(0.0, 1.0) for _ in range(len(combination))]
        test_acc = 0
        weights = [0]*len(combination)
        loss = 0
        weights_j = [0]*len(combination)
        for k in range(count):
            if method == 'differential_evolution':
                weights_j = differential_evolution(evaluate_weights, bounds=bound_w, args=(
                    combination, real_outputs), maxiter=1000, tol=1e-7)['x']
            else:
                weights_j = minimize(evaluate_weights, args=(combination, real_outputs), bounds=bound_w, x0=[
                                     1/len(combination) for _ in range(len(combination))], method=method)['x']
            loss_j = evaluate_weights(weights_j, combination, real_outputs)
            test_acc_j = 1 - \
                evaluate_weights(
                    weights_j, [outputs_test[i] for i in index_combination], real_outputs_test)
            test_acc += test_acc_j
            loss += loss_j
        weights = weights_j
        print(weights)
        print('%d: validation accuracy %f' % (i, (1-loss/count)*100))
        print('previous validation accuracy %f' % ((1-oldLoss)*100))
        print('test performance: %s' % (test_acc*100/count))
        print('previous test performance: %s' % ((1-oldTestLoss)*100))
        model_names = [models[i] for i in index_combination]
        print([models[i] for i in index_combination])

        print()
        results["".join(str(model) + " " for model in model_names)] = {
            "val_acc_1": (1-oldLoss)*100,
            "val_acc_2": (1-loss/count)*100,
            "test_acc_1": ((1-oldTestLoss)*100),
            "test_acc_2": (test_acc*100/count)
        }
        if loss < result:
            result = loss
            best_weights = weights
            best_combinations = index_combination
    return results


if __name__ == '__main__':
    # for method in optimizationMethods:
    for method in ["differential_evolution"]:
        # res = main(evaluate_weights_Borda, method)
        # dump(res, "./votingVars/resBords_"+method+".joblib")
        # res = main(evaluate_weights_Plurarity, method)
        # dump(res, "./votingVars/resPlura_"+method+".joblib")
        res = main(evaluate_weights_kApproval, method)
        dump(res, "./votingVars/resKAppr_"+method+".joblib")
    extractVoting()
    votingSummary()
