# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import tools.DataSet as DataSet

class BasicModel:
    def __init__(self, model=None, **hparams):
        self.hyperparams = hparams
        self.model = model

    def hyperparam_search(self, dataset, folds, **hparams_config):
        folds = dataset.split_in_folds(folds)
        hparams_names = hparams_config.keys()
        hparams_ranges = list(hparams_config.values())
        all_combs = np.array(np.meshgrid(hparams_ranges)).T.reshape(-1, len(hparams_ranges))
        optimal_hparams = self.hyperparams
        min_err = np.inf
        for train, valid in folds:
            for arr in all_combs:
                self.hyperparams = dict(zip(hparams_names, arr))
                self.model.set_params(**self.hyperparams)
                self.model.fit(train.features, train.labels)
                pred = self.model.predict(valid.features)
                err = self.erreur(pred, valid.labels)
                if (min_err > err):
                    min_err = err
                    optimal_hparams = self.hyperparams
        self.hyperparams = optimal_hparams
        print(self.hyperparams)

    def _train(self, dataset, folds=0, **hparams_config):
        if (len(self.hyperparams) != 0 and folds != 0 and len(hparams_config) != 0):
            self.hyperparam_search(dataset, folds, **hparams_config)
        if (len(self.hyperparams) != 0):
            self.model.set_params(**self.hyperparams)
        return self.model.fit(dataset.features, dataset.labels)
    
    def predict(self, features):
        return self.model.predict(features)
    
    def erreur(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)