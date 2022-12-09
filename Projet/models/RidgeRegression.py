# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
from sklearn.linear_model import RidgeClassifier
from models.BasicModel import BasicModel

class RidgeRegression(BasicModel):
    def __init__(self):
        self.model = RidgeClassifier()

    def train(self, features, labels):
        return self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)