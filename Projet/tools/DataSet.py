# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import random

class DataSet:
    """
    Class that contains a set of data along with their respective labels
    """
    def __init__(self, features=np.array([[]]), labels=np.array([])):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return self.labels[index], self.features[index]
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __str__(self):
        return f"Labels : {self.labels}\nFeatures : {self.features}"
    
    def append(self, dataset):
        if (self.labels.size == 0):
            self.features = dataset.features.copy()
            self.labels = dataset.labels.copy()
        else:
            self.features = np.append(self.features, dataset.features, axis=0)
            self.labels = np.append(self.labels, dataset.labels, axis=0)

    def shuffle_data(self):
        zippedData = list(zip(self.features, self.labels))
        random.shuffle(zippedData)
        self.features, self.labels = zip(*zippedData)

    def get_random_samples(self, percentages):
        count_arr = [int(percentages[0] * len(self.labels))]
        for p in percentages[1:]:
            count_arr.append(int((p + count_arr[-1]) * len(self.labels)))
        features = np.array(np.array_split(self.features, count_arr), dtype='object')
        labels = np.array(np.array_split(self.labels, count_arr), dtype='object')
        datasets = [ DataSet(features[i], labels[i]) for i in range(0, len(labels)) ]
        return datasets
    
    def group_by_class(self):
        # Sort based on label idicies
        sorted_labels_idx = self.labels.argsort()
        sorted_labels = self.labels[sorted_labels_idx]
        sorted_features = self.features[sorted_labels_idx]
        # Get the indices where shifts (IDs change) occur
        _, cut_idx = np.unique(sorted_labels, return_index=True)
        # Use the indices to split the input array into sub-arrays with common IDs
        grouped_featured = np.split(sorted_features, cut_idx)[1:]
        grouped_labels = np.split(sorted_labels, cut_idx)[1:]
        # Create array of datasets grouping labels
        grouped_sets = []
        for f, k in zip(grouped_featured, grouped_labels):
            grouped_sets.append(DataSet(f, k))
        return grouped_sets
    
    def split_by_class(self, percentages):
        grouped_datasets = self.group_by_class()
        splits = [ DataSet() for _ in range(len(percentages)+1) ]
        for group in grouped_datasets:
            samples = group.get_random_samples(percentages)
            for i in range(0, len(samples)):
                splits[i].append(samples[i])
        return splits