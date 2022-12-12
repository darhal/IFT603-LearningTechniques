# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.BasicModel import BasicModel


class RandomForest(BasicModel):
    """
    RandomForest classifier inherits from BasicModel
    """

    def __init__(self, stand_trans=False):
        """
        RandomForest constructor

        Inputs : 
            - stand_trans : wether data should be standardised or not
        """
        BasicModel.__init__(
            self,
            RandomForestClassifier(),
            stand_trans, 
        )
