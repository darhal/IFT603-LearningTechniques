# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.svm import SVC
from models.BasicModel import BasicModel

class SupportVectorMachine(BasicModel):
    def __init__(self, kernel="rbf", stand_trans=False, pca_trans=False):
        BasicModel.__init__(self,
            SVC(kernel=kernel, max_iter=3000, C=1.0, probability=True), 
            stand_trans, pca_trans, 
            core_model__C=np.logspace(-6, 1, num=10), 
            core_model__gamma=np.logspace(-6, 1, num=10)
        )