from variables import *
from numpy import argmax
from joblib import load, dump
from numpy import tensordot
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import minimize
import numpy as np
from joblib import dump
from extractVoting import extractVoting, votingSummary
from votingConfig import optimizationMethods
from voting import prepare_data


def assign_vote(votes, weights, n):
    ones = []
    for model, model_weights in zip(votes, weights):
        model_res = []
        for sample in model:
            result = np.zeros_like(sample)
            for i in range(n):
                result[sample.argsort()[-1-i]] = model_weights[i]
            model_res.append(result)
        ones.append(model_res)
    return np.array(ones)


def evaluate_weights_flexible(weights, yhats, y, n):
    weights = np.array(weights).reshape(int(len(weights)/n), n)

    yhats = np.array(yhats)
    yhats = assign_vote(yhats, weights, n)

    summed = tensordot(
        yhats, [1 for _ in range(len(weights))], axes=((0), (0)))
    yhat = argmax(summed, axis=1)

    return 1 - accuracy_score(y, yhat)


def main(method, n):
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
        weights = [1 for _ in range(len(combination)*n)]

        loss = evaluate_weights_flexible(weights, combination, real_outputs, n)
        testLoss = evaluate_weights_flexible(
            weights, [outputs_test[i] for i in index_combination], real_outputs_test, n)
        performances.append((loss, index_combination, combination, testLoss))
    performances.sort(key=lambda tup: tup[0], reverse=False)

    results = {}

    count = 1
    if method == 'differential_evolution':
        count = 5

    for oldLoss, index_combination, combination, oldTestLoss in performances[:50]:
        i += 1
        bound_w = [(1.0, 10.0) for _ in range(len(combination) * n)]
        test_acc = 0
        weights = [0]*len(combination)*n
        loss = 0
        weights_j = [0]*len(combination)*n
        for k in range(count):
            if method == 'differential_evolution':
                weights_j = differential_evolution(evaluate_weights_flexible, bounds=bound_w, args=(
                    combination, real_outputs, n), maxiter=1000, tol=1e-7)['x']
            else:
                weights_j = minimize(evaluate_weights_flexible, args=(combination, real_outputs, n), bounds=bound_w, x0=[
                                     1 for _ in range(len(combination) * n)], method=method)['x']
            loss_j = evaluate_weights_flexible(
                weights_j, combination, real_outputs, n)
            test_acc_j = 1-evaluate_weights_flexible(
                weights_j, [outputs_test[i] for i in index_combination], real_outputs_test, n)
            test_acc += test_acc_j
            loss += loss_j
        weights = weights_j

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
    for method in optimizationMethods:
        res = main(method, 3)
        dump(res, "./votingVars/savedResults/resFlex_"+method+".joblib")
    extractVoting()
    votingSummary()
