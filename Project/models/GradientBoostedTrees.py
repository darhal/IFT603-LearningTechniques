# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from models.BasicModel import BasicModel


class GradientBoostedTrees(BasicModel):
    """
    Gradient Boosted Trees classifier inherits from BasicModel

    Hyper-params :
        - learning_rate : varies from 0.1 to 5 linearly
    """

    def __init__(self, stand_trans=False):
        """
        GradientBoostedTrees constructor

        Inputs :
            - stand_trans : wether data should be standarided or not
        """
        BasicModel.__init__(
            self,
            GradientBoostingClassifier(learning_rate=0.01, probability=True),
            stand_trans,
        )
