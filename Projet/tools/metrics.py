# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import sklearn as sk

def display_performance_metrics(predictions, target, extra_text=""):
    """
    Helper function that display all performance metrics related to predictions and target

    Inputs :
        - predictions : predictions vector (C,)
        - target : ground truth vector (C,)
    """
    confusion_mat, accu, precision, sensitivity, specificity, fallout, f1_score = calculate_performance_metrics(predictions, target)
    sk_accu = sk.metrics.accuracy_score(predictions, target)
    sk_precision, sk_recall, sk_fscore, sk_support = sk.metrics.precision_recall_fscore_support(predictions, target, average='micro')
    print(f"""Performance Metrics {extra_text}:
    Accuracy : {accu} - {sk_accu}
    Precision : {precision} - {sk_precision}
    Sensitivity : {sensitivity} - {sk_recall}
    Specificity : {specificity}
    Fallout : {fallout}
    F1 Score : {f1_score}""")


def calculate_performance_metrics(predictions, target):
    """
    Computes all performance metrics related to given predictions 
    with respect to the ground truth.

    Inputs :
        - predictions : predictions vector (C,)
        - target : ground truth vector (C,)
    Outputs :
        - confusion_mat : confusion matrix (CxC)
        - accu : accuracy (float)
        - precision : precision vector (C,)
        - sensitivity : sensitivty or recall vector (C,)
        - specificity : specificity vector (C,)
        - fallout : fallout vector (C,)
        - f1_score : F1 score (float)
    """
    confusion_mat = confusion_matrix(predictions, target)
    accu, precision, sensitivity, specificity, fallout = confusion_matrix_perf_metrics(confusion_mat)
    f1_score = 2 * ((precision*sensitivity) / (precision+specificity))
    return confusion_mat, accu, precision, sensitivity, specificity, fallout, f1_score


def confusion_matrix(predictions, target):
    """
    Computes the confusion matrix related to given predictions
    with respect to a ground truth vector

    Inputs :
        - predictions : predictions vector (C,)
        - target : ground truth vector (C,)
    Outputs :
        - confusion_mat : confusion matrix (CxC)
    """
    C = len(np.unique(target))
    confusion_mat = np.zeros((C, C), dtype="int32")
    for i in range(len(target)):
        confusion_mat[predictions[i]][target[i]] += 1
    return confusion_mat


def confusion_matrix_perf_metrics(confusion_mat):
    """
    Computes most of the performance metrics of a confusion matrix
    https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

    Inputs :
        - confusion_mat : confusion matrix (CxC)
    Outputs :
        - accu : accuracy (float)
        - precision : precision vector (C,)
        - sensitivity : sensitivty or recall vector (C,)
        - specificity : specificity vector (C,)
        - fallout : fallout vector (C,)
    """
    total = confusion_mat.sum()
    true_pos = np.diag(confusion_mat)
    false_pos = confusion_mat.sum(axis=0) - true_pos
    false_neg = confusion_mat.sum(axis=1) - true_pos
    true_neg = total - (false_pos + false_neg + true_pos)
    true_pos = true_pos.sum()
    false_pos = false_pos.sum()
    false_neg = false_neg.sum()
    true_neg = true_neg.sum() 
    # Precision or positive predictive value
    precision = true_pos / (true_pos+false_pos)
    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = true_pos / (true_pos+false_neg)
    # Specificity or true negative rate
    specificity = true_neg / (true_neg+false_pos)
    # Fall out or false positive rate
    fallout = false_pos / (false_pos+true_neg)
    # Accurcy
    accu = true_pos / total
    return accu, precision, sensitivity, specificity, fallout
