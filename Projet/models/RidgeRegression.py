# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import RidgeClassifier
from models.BasicModel import BasicModel

class RidgeRegression(BasicModel):
    def __init__(self):
        BasicModel.__init__(self, RidgeClassifier(), alpha=1.0)
    
    def train(self, dataset):
        return BasicModel._train(self, dataset, 10, alpha=np.logspace(-9, 0.3, num=10))