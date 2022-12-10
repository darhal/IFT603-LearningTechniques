# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import numpy as np
from sklearn.linear_model import Perceptron
from models.BasicModel import BasicModel

class SinglePerceptron(BasicModel):
    def __init__(self):
        BasicModel.__init__(self, Perceptron(eta0=0.001, penalty='l2'), alpha=1.0)
    
    def train(self, dataset):
        return BasicModel._train(self, dataset, 10, alpha=np.logspace(-9, 0.3, num=10))