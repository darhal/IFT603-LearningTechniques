# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel


class LogisticClassifier(BasicModel):
    """
    Logistic regression classifier inherits from BasicModel

    Hyper-params:
        - C : varies from 10e-7 to 10e1 logarthimically
    """
    
    def __init__(self, stand_trans=False):
        """
        LogisticClassifier constructor

        Inputs : 
            - stand_trans : wether data should be standarided or not
        """
        BasicModel.__init__(
            self,
            LogisticRegression(max_iter=3000),
            stand_trans,
            core_model__C=np.logspace(-7, 1, num=20)
        )
