# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.neural_network import MLPClassifier
from models.BasicModel import BasicModel

class MultiLayerPerceptron(BasicModel):
    def __init__(self, stand_trans=False, pca_trans=False):
        BasicModel.__init__(self,
            MLPClassifier(hidden_layer_sizes=(200, 150, 100), activation='relu', solver='adam',
                alpha=0.0001, batch_size='auto', max_iter=200),
            stand_trans, pca_trans
        )
