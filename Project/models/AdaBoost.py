# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel


class AdaBoost(BasicModel):
    """
    Ada Boost Classifier classifier inherits from BasicModel

    Hyper-params :
        - learning_rate : varies from 0.1 to 5 linearly
    """

    def __init__(self, stand_trans=False):
        """
        AdaBoost constructor

        Inputs :
            - stand_trans : wether data should be standardised or not
        """
        BasicModel.__init__(
            self,
            AdaBoostClassifier(),
            stand_trans,
            core_model__learning_rate=np.linspace(0.1, 10.0, num=5),
            core_model__n_estimators=np.linspace(
                10, 200, num=5, dtype=np.int32),
        )
