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
    """
    ModelTester a helper class that takes model class name and create all possible variations of it.
    Helps tests the same model with different configurations easily.

    Fields : 
        - class_name : model class name
        - model_configs : different configs of the model to load
        - self.models : array of different variations of the same model
    """

    def __init__(self, class_name):
        """
        ModelTester constructor

        Inputs : 
            - class_name : name of the class to load
        """
        module = importlib.import_module(f"models.{class_name}")
        model_class = getattr(module, class_name)
        self.class_name = class_name
        self.model_configs = [False, True]
        self.models = [model_class(stand_trans=config) for config in self.model_configs]

    def train(self, train_set, show_graph=True):
        """
        Trains all models while measuring their train performance and hyperparams curve.

        Inputs : 
            - train_set : dataset containing the train set
            - show_graph : boolean indicating if hparams curve should be shown
        Output : 
            - panda data frame containing perf metrics for different models
        """
        perf_matrix = []
        components = self.models[0].hparams_config.keys()
        show_plot = len(self.models[0].hparams_config) and show_graph
        if (show_plot):
            fig, ax = plt.subplots(len(components), len(self.models), figsize=(14, 8), dpi=120)
            fig.supylabel("Classification accuracy")
            fig.supxlabel(f"Hyper-param curve for \n{self.class_name}")
        for ax_idx, model in enumerate(self.models):
            model.train(train_set)
            if (show_plot):
                model.visualise_hyperparam_curve(
                    ax[:,ax_idx] if len(components) > 1 else [ax[ax_idx]],
                    title=f"Stand={self.model_configs[ax_idx]}"
                )
            probs, classes = model.predict_probs(train_set.features)
            perf_matrix.append(get_performance_metrics(classes, train_set.labels, probs))
        if show_plot:
            fig.tight_layout()
            plt.show()
        return self.build_perf_dataframe(perf_matrix, "TRAIN")

    def predict(self, test_set):
        """
        Run all models predict while measuring their performance.

        Inputs : 
            test_set : test dataset
         Output : 
            - panda data frame containing perf metrics for different models
        """
        perf_matrix = []
        for i, m in enumerate(self.models):
            probs, classes = m.predict_probs(test_set.features)
            perf_matrix.append(get_performance_metrics(classes, test_set.labels, probs))
        return self.build_perf_dataframe(perf_matrix, "TEST")

    def build_perf_dataframe(self, perf_matrix, title=""):
        """
        Helper function that builds panda dataframe out of the perf_matrix

        Inputs : 
            - perf_matrix : performance matrix containing perf metrics per column
            - title : extra title string
        Output : 
            - panda data frame containing perf metrics for different models
        """
        return pd.DataFrame(
            data=np.array(perf_matrix).T,
            columns=[f"[{title}] {self.class_name} (Stand={config})" for config in self.model_configs],
            index=["Accuracy", "Precision", "Recall",
                   "F1 Score", "ROC AUC", "Log loss"]
        )

    def visualise_learning_curve(self, dataset):
        """
        Helper function that shows the learning curve graph of the different models on the entire data set.
        Taking only a portion of the dataset at a time using random stratified selection.

        Inputs : 
            - dataset: the entirety of the dataset
        Outputs : void
        """
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
        fig, ax = plt.subplots(1, len(self.models), figsize=(14, 8), dpi=120)
        for ax_idx, estimator in enumerate(self.models):
            LearningCurveDisplay.from_estimator(estimator.model, **common_params, ax=ax[ax_idx])
            handles, label = ax[ax_idx].get_legend_handles_labels()
            ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
            ax[ax_idx].set_title(
                f"Learning Curve for {self.class_name}\n(Stand={self.model_configs[ax_idx]})"
            )
        plt.show()

    def test(self, dataset, train_set, test_set, show_learning_curve=True):
        """
        A helper functions that perform all other functiosnt hat help asses the performance of a model.
        It train models on the train_set and shows it's train performance.
        Then it tests the models on the test_set and shows it's test performance.
        Then it show the learning_curve of the models on the entire dataset.

        Inputs : 
            - dataset : entire dataset
            - train_set: train set
            - test_set : test set
            - show_learning_curve : display or not the learning curve
        Outputs : void
        """
        if (show_learning_curve):
            self.visualise_learning_curve(dataset)
        train_perf = self.train(train_set)
        test_perf = self.predict(test_set)
        combined_pd = pd.concat([train_perf.iloc[:,0], test_perf.iloc[:,0], train_perf.iloc[:,1], test_perf.iloc[:,1]], axis=1)
        display(combined_pd)
