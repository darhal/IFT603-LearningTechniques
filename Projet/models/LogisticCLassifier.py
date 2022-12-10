# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel

class LogisticClassifier(BasicModel):
    def __init__(self, norm_trans=False, pca_trans=False):
        BasicModel.__init__(self, LogisticRegression(max_iter=1000), norm_trans, pca_trans, core_model__C=1.0)
    
    def train(self, dataset):
        return BasicModel._train(self, dataset, core_model__C=np.logspace(-9, 0.3, num=20))
