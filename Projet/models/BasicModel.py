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
    def __init__(self, core_model=None, norm_trans=False, pca_trans=False, **hparams):
        pipe_steps = []
        if (norm_trans):
            pipe_steps.append(("scaler", StandardScaler()))
        if (pca_trans):
            pipe_steps.append(("pca", PCA()))
        pipe_steps.append(("core_model", core_model))
        self.pipe = Pipeline(steps=pipe_steps)
        self.model = self.pipe
        self.model.set_params(**hparams)
        self.hyperparams = hparams
    
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

    def _train(self, dataset, folds=6, **hparams_config):
        if (len(self.hyperparams) != 0 and folds != 0 and len(hparams_config) != 0):
            self.model = GridSearchCV(self.pipe, hparams_config, cv=StratifiedKFold(folds, shuffle=True), n_jobs=4, return_train_score=True)
        return self.model.fit(dataset.features, dataset.labels)

    def visualise_train_perf(self, axes):
        if len(self.hyperparams) == 0:
            return
        results = pd.DataFrame(self.model.cv_results_)
        components = list(self.hyperparams.keys())
        for i in range(0, len(self.hyperparams.keys())):
            components_col = f"param_{components[i]}"
            best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, "mean_test_score"))
            best_clfs.plot(x=components_col, y="mean_test_score", legend=False, ax=axes[i])
            axes[i].set_ylabel("Classification accuracy")
            axes[i].set_xlabel(components[i])
    