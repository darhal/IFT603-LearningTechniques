# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import sys
from tools.DataLoader import DataLoader
from models.LinearRegression import LinearRegression
from tools.Metrics import *

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
print(len(train_set.group_by_class()))
print(len(test_set.group_by_class()))
print(len(train_set))
print(len(test_set))

# Linear Regression : 
lr = LinearRegression()
lr.train(train_set.features, train_set.labels)
print(train_set.features)
pred_train = lr.predict(train_set.features)
pred_test = lr.predict(test_set.features)

confusion_mat, accu, precision, sensitivity, specificity, fallout, f1_score = calculate_performance_metrics(pred_train, train_set.labels)
print(f"""
Accuracy : {accu}
Precision : {precision.mean()}
Sensitivity : {sensitivity.mean()}
Specificity : {specificity.mean()}
Fallout : {fallout.mean()}
F1 Score : {f1_score}
""")

confusion_mat, accu, precision, sensitivity, specificity, fallout, f1_score = calculate_performance_metrics(pred_test, test_set.labels)
print(f"""
Accuracy : {accu}
Precision : {precision.mean()}
Sensitivity : {sensitivity.mean()}
Specificity : {specificity.mean()}
Fallout : {fallout.mean()}
F1 Score : {f1_score}
""")

