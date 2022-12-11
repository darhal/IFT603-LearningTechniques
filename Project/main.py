# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import sys
import numpy as np
import sklearn as sk
from tools.DataLoader import DataLoader
from tools.ModelTester import ModelTester
from models import LinearModels
from tools.Metrics import *

from models.RidgeRegression import RidgeRegression


dl = DataLoader("data/train.csv", class_col_name="species", excluded_features={"id"})
dl.load()
dataset = dl.get_dataset()
# print(len(dataset))

#for ds in dataset.group_by_class():
#    print(len(ds))

#print(len(dataset.group_by_class()))
#print("---")

#for i in dataset.get_random_samples([0.1]):
#    print(len(i))

train_set, test_set = dataset.split_by_class([0.9])
#print(len(train_set.group_by_class()))
#print(len(test_set.group_by_class()))
#print(len(train_set))
#print(len(test_set))

def measure_model_performance(model, train_set, test_set):
    model.train(train_set)
    pred_train = model.predict(train_set.features)
    pred_test = model.predict(test_set.features)
    print(f"~~~~~~~~~~~~~~~ {type(model)} ~~~~~~~~~~~~~~~")
    display_performance_metrics(pred_train, train_set.labels, "(Train data)")
    display_performance_metrics(pred_test, test_set.labels, "(Test data)")

# Linear Regression : 
mt = ModelTester("LinearRegression")
mt.test(train_set, test_set)

# Ridge Regression : 
mt = ModelTester("RidgeRegression", core_model__alpha=np.logspace(-9, 0.3, num=20))
mt.test(train_set, test_set)
#measure_model_performance(LinearModels.RidgeRegression(), train_set, test_set)

# Logistic Regression :
mt = ModelTester("LogisticClassifier", core_model__C=np.logspace(-9, 0.3, num=20))
mt.test(train_set, test_set)

# Perceptron : 
mt = ModelTester("SinglePerceptron", core_model__alpha=np.logspace(-9, 0.3, num=20))
mt.test(train_set, test_set)