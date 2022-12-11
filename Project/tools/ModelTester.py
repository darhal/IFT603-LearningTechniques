# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tools.Metrics import *

class ModelTester():
    def __init__(self, class_name, **hparams_config):
        module = importlib.import_module(f"models.{class_name}")
        model_class = getattr(module, class_name)
        self.class_name = class_name
        self.hparams_config = hparams_config
        self.model_configs = [ (False, False), (True, False), (True, True) ]
        self.models = [ model_class(norm_trans=config[0], pca_trans=config[1]) for config in self.model_configs ]

    def train(self, train_set, show_graph=True):
        perf_matrix = []
        components = self.hparams_config.keys()
        show_plot = len(self.hparams_config) and show_graph 
        if (show_plot): fig, ax = plt.subplots(len(components), len(self.models), figsize=(14,6), dpi=120)
        for i in range(len(self.models)):
            self.models[i].train(train_set)
            if (show_plot): self.models[i].visualise_train_perf(ax[:,i] if len(components) > 1 else [ax[i]])
            probs, classes = self.models[i].predict_probs(train_set.features)
            perf_matrix.append(get_performance_metrics(classes, train_set.labels, probs))
        if show_plot: plt.show()
        return self.build_perf_dataframe(perf_matrix)
    
    def predict(self, test_set):
        perf_matrix = []
        for m in self.models:
            probs, classes = m.predict_probs(test_set.features)
            perf_matrix.append(get_performance_metrics(classes, test_set.labels, probs))
        return self.build_perf_dataframe(perf_matrix)

    def build_perf_dataframe(self, perf_matrix):
        return pd.DataFrame(
            data=np.array(perf_matrix).T,
            columns=[f"{self.class_name} (Norm={config[0]},PCA={config[1]})" for config in self.model_configs],
            index=["Log loss", "Accuracy", "Precision", "Sensitivity", "Specificity", "Fallout", "F1 Score"]
        )
    
    def test(self, train_set, test_set):
        print(f"~~~~~~~~~~~~~~~ TRAIN SET ~~~~~~~~~~~~~~~")
        display(self.train(train_set))
        print(f"~~~~~~~~~~~~~~~ TEST SET ~~~~~~~~~~~~~~~")
        display(self.predict(test_set))
    