from itertools import chain, combinations
from keras.utils import to_categorical
import numpy as np
import itertools
import scipy.special
from time import time


def powerset(iterable, order=None):
    if order is None:
        order = len(iterable)

    return {r: list(combinations(iterable, r)) for r in range(order + 1)}


def convert_index_to_categorical(indices, d):
    a, b = indices.shape
    if b == 0:
        return np.zeros((a, d))
    else:
        indices = np.reshape(indices, [-1])
        cats = to_categorical(indices, num_classes=d)
        cats = np.reshape(cats, (a, b, d))
        return np.sum(cats, axis=-2)


def construct_positions_connectedshapley(d, k, max_order=None):
    subsets = {}

    while k >= d:
        k -= 2

    if max_order == None:
        max_order = k + 1

    coefficients = {j: 2.0 / (j * (j + 1) * (j + 2)) for j in range(1, max_order + 1)}

    for i in range(d):
        subsets[i] = [np.array(range(i - s, i + t + 1)) % d for s in range(int(k / 2 + 1)) for t in
                      range(int(k / 2 + 1))]
        subsets[i] = [subset for subset in subsets[i] if len(subset) <= max_order]

    # Construct dictionary of indices for points where
    # the predict is to be evaluated.
    positions_dict = {(i, l, fill): [] for i in range(d) for l in range(1, max_order + 1) for fill in [0, 1]}
    for i in subsets:
        # pad positions outside S by 1.
        one_pads = 1 - np.sum(to_categorical(np.array(range(int(i - k / 2), int(i + k / 2 + 1))) % d, num_classes=d),
                              axis=0)

        for arr in subsets[i]:
            # For feature i, the list of subsets of size j, with/without feature i.
            l = len(arr)
            pos_included = np.sum(to_categorical(arr, num_classes=d), axis=0)
            # pad positions outside S by 1.

            pos_excluded = pos_included - to_categorical(i, num_classes=d)
            positions_dict[(i, l, 1)].append(pos_included)
            positions_dict[(i, l, 0)].append(pos_excluded)

    # values is a list of lists of zero-one vectors.
    keys, values = positions_dict.keys(), positions_dict.values()

    # concatenate a list of lists to a list.

    values = [np.array(value).reshape(-1, d) for value in values]
    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    positions, unique_inverse = np.unique(positions, axis=0, return_inverse=True)

    return positions_dict, key_to_idx, positions, coefficients, unique_inverse


def explain_shapley(predict, x, d, k, positions, key_to_idx, inputs, coefficients, unique_inverse):
    """
    Compute the importance score of each feature of x for the predict.

    Inputs:
    predict: a function that takes in inputs of shape (n,d)

    x: input vector (d,)

    k: number of neighbors taken into account for each feature.

    Outputs:
    phis: importance scores of shape (d,)
    """
    while k >= d:
        k -= 2
    st1 = time()

    f_vals = predict(inputs)

    probs = predict(np.array([x]))
    st2 = time()
    log_probs = np.log(f_vals + np.finfo(float).resolution)
    phis_dict = []

    discrete_probs = np.eye(len(probs[0]))[np.argmax(probs, axis=-1)]
    # print(discrete_probs.shape)
    vals = (probs * np.exp(log_probs))

    key_to_val = {key: np.array([vals[unique_inverse[idx]] for idx in key_to_idx[key]]) for key in key_to_idx}
    # print(key_to_val)
    phis = []
    for i in range(d):
        xx = []
        for j in coefficients:
            dif = key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)]
            if len(dif) != 0:
                xx.append(np.sum((coefficients[j]) * (dif), axis=0))
        xx = np.array(xx)
        phis.append(np.sum(xx, axis=0))

    return phis


def text_shapley_(dataset, args):
    st = time()
    print('Making explanations...')
    metric = 'prob'
    scores = []

    predict = dataset.func
    method = 'connectedshapley'

    select = True

    construct_positions_dict = lambda d, k, select: construct_positions_connectedshapley(d, k, args.max_order)

    for i, sample in enumerate(dataset.x_val):
        print('explaining the {}th sample...'.format(i))
        # Evaluate model at inputs.

        d = dataset.val_len
        positions_dict, key_to_idx, positions, coefficients, unique_inverse = construct_positions_dict(d,
                                                                                                       args.num_neighbors,
                                                                                                       select)
        sample = sample[-d:]
        inputs = sample * positions

        score = explain_shapley(predict, sample, d, args.num_neighbors, positions, key_to_idx, inputs, coefficients,
                                unique_inverse)

        # for metric in metrics:
        scores.append(score)

    scores = np.array(scores)

    # save data.

    print('Time spent is {}'.format(time() - st))
    scores = scores.T
    scores = ([score.T for score in scores])
    return (scores)





