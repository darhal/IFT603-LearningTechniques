# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np

class DataSet:
    """
    Class that contains a set of data along with their respective labels
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return self.labels[index], self.features[index]
    
    def __len__(self):
        return len(self.labels)
