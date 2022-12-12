# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.neural_network import MLPClassifier
from models.BasicModel import BasicModel

class MultiLayerPerceptron(BasicModel):
    """
    MultiLayerPerceptron class inherits from BasicModel and implements a multi preceptron network
    """

    def __init__(self, stand_trans=False):
        """
        MultiLayerPerceptron constructor

        Inputs : 
            - stand_trans : wether data should be standarided or not
        """
        BasicModel.__init__(self,
            MLPClassifier(
                hidden_layer_sizes=(196,), 
                activation='relu', 
                solver='lbfgs',
                alpha=0.001, 
                batch_size='auto', 
                max_iter=300,
                early_stopping=True,
                learning_rate='adaptive'
            ),
            stand_trans
        )
