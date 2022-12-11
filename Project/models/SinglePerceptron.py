# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import Perceptron
from models.BasicModel import BasicModel

class SinglePerceptron(BasicModel):
    def __init__(self, stand_trans=False, pca_trans=False):
        BasicModel.__init__(self,
            Perceptron(eta0=0.001, penalty='l2', max_iter=3000), 
            stand_trans, pca_trans, 
            core_model__alpha=np.logspace(-9, 0.5, num=10)
        )