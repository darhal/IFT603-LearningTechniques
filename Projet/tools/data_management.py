# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import csv
import numpy as np

class DataManagement:
    def __init__(self, type, model, train_data, test_data):
        self.type = type
        self.model = model
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.x_train = None
        self.t_train = None
        self.x_test = None
        # self.t_test = None  # not provided

    def load_data(self):
        # Load training dataset, split t flags and x data
        if type(self.train_dataset) == type.__str__:
            self.x_train = None
            self.t_train = None
            with open(self.train_dataset, newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                next(reader)
                for row in reader:
                    if not self.x_train:
                        self.x_train = np.array(row[2:], float).reshape(1, len(row[2:]))
                    else:
                        self.x_train = np.concatenate((self.x_train, np.array(row[2:], float).reshape(1, len(row[2:]))))
                    if not self.t_train:
                        self.t_train = np.array(row[1]).reshape(1, 1)
                    else:
                        self.t_train = np.concatenate((self.t_train, np.array(row[2:], float)))
        # Load testing dataset, split t flags and x data
        if type(self.test_dataset) == type.__str__:
            self.x_test = None
            with open(self.train_dataset, newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                next(reader)
                for row in reader:
                    if not self.x_train:
                        self.x_test = np.array(row[1:], float).reshape(1, len(row[1:]))
                    else:
                        self.x_test = np.concatenate(self.x_test, np.array(row[1:], float).reshape(1, len(row[1:])))

    def group_data_by_class():  # necessary ?
        ...