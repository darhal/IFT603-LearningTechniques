# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

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
    def __init__(self, core_model=None, stand_trans=False, pca_trans=False, **hparams_config):
        pipe_steps = []
        if (stand_trans):
            pipe_steps.append(("scaler", StandardScaler()))
        if (pca_trans):
            pipe_steps.append(("pca", PCA()))
        pipe_steps.append(("core_model", core_model))
        self.pipe = Pipeline(steps=pipe_steps)
        self.model = self.pipe
        self.hparams_config = hparams_config
    
    def predict(self, features):
        return self.model.predict(features)

    def predict_probs(self, features):
        if (hasattr(self.model, 'predict_proba')):
            probs = self.model.predict_proba(features)
            classes = np.argmax(probs, axis=1)
        else:
            probs = None
            classes = self.predict(features)
        return probs, classes
    
    def erreur(self, predictions, labels):
        return np.mean(np.abs(predictions - labels))

    def train(self, dataset, folds=5):
        if (len(self.hparams_config) != 0 and folds != 0):
            self.model = GridSearchCV(
                self.pipe, 
                self.hparams_config, 
                cv=StratifiedKFold(folds, shuffle=True), 
                n_jobs=4, 
                return_train_score=True
            )
        return self.model.fit(dataset.features, dataset.labels)

    def visualise_hyperparam_curve(self, axes, title=""):
        if len(self.hparams_config) == 0:
            return
        results = pd.DataFrame(self.model.cv_results_)
        components = list(self.hparams_config.keys())
        for i in range(0, len(self.hparams_config.keys())):
            components_col = f"param_{components[i]}"
            best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, "mean_test_score"))
            best_clfs.plot(x=components_col, y="mean_test_score", legend=False, ax=axes[i])
            axes[i].set_xlabel(components[i])
            axes[i].set_title(f"{title}\n{components[i]}")
    