# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
#####

import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from tools.Metrics import *
from sklearn.model_selection import LearningCurveDisplay

class ModelTester():
    def __init__(self, class_name):
        module = importlib.import_module(f"models.{class_name}")
        model_class = getattr(module, class_name)
        self.class_name = class_name
        self.model_configs = [ (False, False), (True, False), (True, True) ]
        self.models = [ model_class(stand_trans=config[0], pca_trans=config[1]) for config in self.model_configs ]

    def train(self, train_set, show_graph=True):
        perf_matrix = []
        components = self.models[0].hparams_config.keys()
        show_plot = len(self.models[0].hparams_config) and show_graph 
        if (show_plot): 
            fig, ax = plt.subplots(len(components), len(self.models), figsize=(14,8), dpi=120)
            fig.supylabel("Classification accuracy")
        for ax_idx, model in enumerate(self.models):
            model.train(train_set)
            if (show_plot): 
                model.visualise_hyperparam_curve(
                    ax[:,ax_idx] if len(components) > 1 else [ax[ax_idx]],
                    title=f"Hyper-param curve for \n{self.class_name}(Stand={self.model_configs[ax_idx][0]},PCA={self.model_configs[ax_idx][1]})"
                )
            probs, classes = model.predict_probs(train_set.features)
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
            columns=[f"{self.class_name} (Stand={config[0]},PCA={config[1]})" for config in self.model_configs],
            index=["Accuracy", "Precision", "Sensitivity", "Specificity", "Fallout", "F1 Score", "Log loss"]
        )
    
    def visualise_learning_curve(self, dataset):
        common_params = {
            "X": dataset.features,
            "y": dataset.labels,
            "train_sizes": np.linspace(0.1, 1.0, 10),
            "cv": sk.model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0),
            "score_type": "both",
            "n_jobs": 4,
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
            "score_name": "Accuracy",
        }
        fig, ax = plt.subplots(1, len(self.models), figsize=(14,8), dpi=120)
        for ax_idx, estimator in enumerate(self.models):
            LearningCurveDisplay.from_estimator(estimator.model, **common_params, ax=ax[ax_idx])
            handles, label = ax[ax_idx].get_legend_handles_labels()
            ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
            ax[ax_idx].set_title(
                f"Learning Curve for {self.class_name}\n(Stand={self.model_configs[ax_idx][0]},PCA={self.model_configs[ax_idx][1]})"
            )
        plt.show()

    def test(self, dataset, train_set, test_set):
        print(f"~~~~~~~~~~~~~~~ Learning curve ~~~~~~~~~~~~~~~")
        self.visualise_learning_curve(dataset)
        print(f"~~~~~~~~~~~~~~~ TRAIN SET ~~~~~~~~~~~~~~~")
        display(self.train(train_set))
        print(f"~~~~~~~~~~~~~~~ TEST SET ~~~~~~~~~~~~~~~")
        display(self.predict(test_set))
    