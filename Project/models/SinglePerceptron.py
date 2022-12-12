# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import Perceptron
from models.BasicModel import BasicModel


class SinglePerceptron(BasicModel):
    """
    SinglePerceptron classifier inherits from BasicModel

    Hyper-params : 
        - alpha : varies from 10e-9 to 10e0.5 lograthimically
    """

    def __init__(self, stand_trans=False):
        """
        SinglePerceptron constructor

        Inputs : 
            - stand_trans : wether data should be standardised or not
        """
        BasicModel.__init__(
            self,
            Perceptron(eta0=0.001, penalty='l2', max_iter=3000),
            stand_trans,
            core_model__alpha=np.logspace(-7, 0.5, num=20)
        )
