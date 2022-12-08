# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import matplotlib as plt


def stats_by_class(x_data, y_data):
    """
    Compute statistics for each class.
    x_data: values mapped to y_data elements
    y_data: names of group (name of classes)
    """
    # Group data by class
    data_by_class = {}
    for x, y in zip(x_data, y_data):
        if y not in data_by_class:
            data_by_class[y] = np.array(y.reshape(1, len(y)))  # can be optimized
        else:
            np.concatenate((data_by_class[y], np.array(x).reshape(1, len(x))))

    stats_by_class = {}
    for c, data in data_by_class.items():
        mean = data.mean()
        median = data.median()
        std = data.std()
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        stats_by_class.push(c, {"size": len(data), "mean": mean, "median": median,
                            "std": std, "Q1": q1, "Q3": q3})
    return stats_by_class

def visualization(method, data):
    """
    Visualize a view of data of the class using a method.
    """
    ...
