# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel

class LogisticClassifier(BasicModel):
    def __init__(self):
        BasicModel.__init__(self, LogisticRegression(), C=1.0)
    
    def train(self, dataset):
        return BasicModel._train(self, dataset, 10, C=np.logspace(10e-6, 1.0, num=10))
