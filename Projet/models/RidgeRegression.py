# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import RidgeClassifier
from models.BasicModel import BasicModel

class RidgeRegression(BasicModel):
    def __init__(self, norm_trans=False, pca_trans=False):
        BasicModel.__init__(self, RidgeClassifier(max_iter=1000), norm_trans, pca_trans, core_model__alpha=1.0)
    
    def train(self, dataset):
        return BasicModel._train(self, dataset, core_model__alpha=np.logspace(-9, 0.3, num=20))