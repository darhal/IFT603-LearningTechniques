# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tools.DataSet as DataSet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


class BasicModel:
    """
    BasicModel class provides basic helper and utility functions faciliating
    the creation of new models.

    Fields :
        - pipe : containing SK pipeline of different stages of the model
        - model : contains the final SK model (could be after grid search if hparams are specified)
        - hparams_config : dictionary containing hparams and their search space (could be empty)
    """

    def __init__(self, core_model=None, stand_trans=False, **hparams_config):
        """
        BasicModel constructor

        Inputs:
            - core_model : core SK estimator
            - stand_trans : boolean indicating wether standardization should be added to the pipeline
            - hparams_config: variadic paramter which specify hyper paramters and their search space
        """
        pipe_steps = []
        if stand_trans:
            pipe_steps.append(("scaler", StandardScaler()))
        pipe_steps.append(("core_model", core_model))
        self.pipe = Pipeline(steps=pipe_steps)
        self.model = self.pipe
        self.hparams_config = hparams_config

    def train(self, dataset, folds=5):
        """
        Initiate the train of the whole pipeline.
        If hparams are specificed then hparams search phase will be excuted.

        Inputs :
            - dataset : dataset that contains train data
            - folds : number of folds to use for cross-validation
        """
        if len(self.hparams_config) != 0 and folds != 0:
            self.model = GridSearchCV(
                self.pipe,
                self.hparams_config,
                cv=StratifiedKFold(folds, shuffle=True),
                n_jobs=4,
                return_train_score=True,
            )
        return self.model.fit(dataset.features, dataset.labels)

    def predict(self, features):
        """
        Predict new input based on what we have learned previously

        Inputs :
            - features : (N,M) matrix containing features of the elements we want to predict
        Output :
            - labels : (N,) integer vector containing class ID
        """
        return self.model.predict(features)

    def predict_probs(self, features):
        """
        Return probability predictions and predicted classes.
        if there is no predict_proba function supported by the core model,
        then the probability array will be None.

        Inputs :
            - features : (N,M) matrix containing features of the elements we want to predict
        Output :
            - probs : (N,M) float matrix containing probabilities for each class
            - labels : (N,) integer vector containing class ID
        """
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)
            classes = np.argmax(probs, axis=1)
        else:
            probs = None
            classes = self.predict(features)
        return probs, classes

    def erreur(self, predictions, labels):
        """
        Function that returns mean square erreur.

        Inputs :
            - predictions : (N,) integer vector of predicted classes
            - labels : (N,) integer vector of ground truth classes
        """
        return np.mean((predictions - labels) ** 2)

    def visualise_hyperparam_curve(self, axis, title=""):
        """
        Helper function that visualise hyper param curve and the evolution of accuracy.

        Inputs :
            - axis : matplotlib axis
            - title : custom title for the subgraph
        """
        if len(self.hparams_config) == 0:
            return
        results = pd.DataFrame(self.model.cv_results_)
        components = list(self.hparams_config.keys())
        for i in range(0, len(self.hparams_config.keys())):
            components_col = f"param_{components[i]}"
            best_clfs = results.groupby(components_col).apply(
                lambda g: g.nlargest(1, "mean_test_score")
            )
            best_clfs.plot(
                x=components_col, y="mean_test_score", legend=False, ax=axis[i]
            )
            axis[i].set_xlabel(components[i])
            axis[i].set_title(f"{title}")
