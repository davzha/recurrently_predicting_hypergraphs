"""
Adapted from https://github.com/hadarser/SetToGraphPaper/blob/master/performance_eval/eval_test_jets.py (Apache-2.0 License)
"""

import torch
import uproot
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn.metrics

def _get_rand_index(labels, predictions):
    n_items = len(labels)
    if (n_items < 2):
        return 1
    n_pairs = (n_items * (n_items - 1)) / 2

    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs += 1

    return correct_pairs / n_pairs

def _error_count(labels, predictions):
    n_items = len(labels)

    true_positives = 0
    false_positive = 0
    false_negative = 0

    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true):
                true_positives += 1
            if (not label_true) and pred_true:
                false_positive += 1
            if label_true and (not pred_true):
                false_negative += 1
    return true_positives, false_positive, false_negative


def _get_recall(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_negative == 0:
        return 0

    return true_positives / (true_positives + false_negative)


def _get_precision(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_positive == 0:
        return 0
    return true_positives / (true_positives + false_positive)


def _f_measure(labels, predictions):
    precision = _get_precision(labels, predictions)
    recall = _get_recall(labels, predictions)

    if precision == 0 or recall == 0:
        return 0

    return 2 * (precision * recall) / (recall + precision)


def test_performance(model, ds):

    pred = _predict_on_test_set(model, ds)

    test_ds = uproot.open(ds.filename)
    jet_df = test_ds['tree'].pandas.df(['jet_flav', 'trk_vtx_index'], flatten=False)
    jet_flav = jet_df['jet_flav']

    target = [x for x in jet_df['trk_vtx_index'].values]

    print('Calculating scores on test set... ', end='')
    start = datetime.now()
    model_scores = {}
    model_scores['RI'] = np.vectorize(_get_rand_index)(target, pred)
    model_scores['ARI'] = np.vectorize(sklearn.metrics.adjusted_rand_score)(target, pred)
    model_scores['P'] = np.vectorize(_get_precision)(target, pred)
    model_scores['R'] = np.vectorize(_get_recall)(target, pred)
    model_scores['F1'] = np.vectorize(_f_measure)(target, pred)

    end = datetime.now()
    print(f': {str(end - start).split(".")[0]}')

    flavours = {5: 'b jets', 4: 'c jets', 0: 'light jets'}
    metrics_to_table = ['P', 'R', 'F1', "RI", "ARI"]

    df = pd.DataFrame(index=flavours.values(), columns=metrics_to_table)

    for flav_n, flav in flavours.items():
        for metric in metrics_to_table:
            mean_metric = np.mean(model_scores[metric][jet_flav == flav_n])
            df.at[flav, metric] = mean_metric

    return df


def _predict_on_test_set(model, ds):
    model.eval()

    n_tracks = np.array([ds[i][0].shape[0] for i in range(len(ds))])

    indx_list = []
    predictions = []

    for tracks_in_jet in range(2, np.amax(n_tracks)+1):
        trk_indxs = np.where(n_tracks == tracks_in_jet)[0]
        if len(trk_indxs) < 1:
            continue
        indx_list += list(trk_indxs)

        input_batch = torch.stack([ds[i][0] for i in trk_indxs])  # shape (B, N_i, 10)
        
        for i in range(input_batch.size(0)//1024+1):
            with torch.no_grad():
                pred_inc = model.forward(input_batch[i*1024:(i+1)*1024])[...,:-1]
                pred = pred_inc.argmax(1)
                predictions.extend(list(pred.cpu().numpy()))

    sorted_predictions = [x for _, x in sorted(zip(indx_list, predictions))]
    return sorted_predictions