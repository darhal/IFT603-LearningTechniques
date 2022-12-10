# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import LogisticRegression
from models.BasicModel import BasicModel

class LinearRegression(BasicModel):
    def __init__(self):
        BasicModel.__init__(self, LogisticRegression())

    def train(self, dataset):
        return BasicModel._train(self, dataset)