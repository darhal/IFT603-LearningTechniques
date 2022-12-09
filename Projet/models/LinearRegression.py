# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel

class LinearRegression(BasicModel):
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, features, labels):
        return self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)