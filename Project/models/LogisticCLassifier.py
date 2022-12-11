# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel

class LogisticClassifier(BasicModel):
    def __init__(self, stand_trans=False, pca_trans=False):
        BasicModel.__init__(self, 
            LogisticRegression(max_iter=1000), 
            stand_trans, 
            pca_trans, 
            core_model__C=np.logspace(-6, 1, num=10)
        )
