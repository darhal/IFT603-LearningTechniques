# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import RidgeClassifier
from models.BasicModel import BasicModel


class RidgeRegression(BasicModel):
    """
    RidgeRegression classifier inherits from BasicModel

    Hyper-params :
        - alpha : varies logarthimically between 10e-9 and 10e0.5
    """

    def __init__(self, stand_trans=False):
        """
        RidgeRegression constructor

        Inputs :
            - stand_trans : wether data should be standardised or not
        """
        BasicModel.__init__(
            self,
            RidgeClassifier(max_iter=1000),
            stand_trans,
            core_model__alpha=np.logspace(-7, 0.5, num=20),
        )
