# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import sys
import tools.metrics as mt
import tools.data_management as dm


def main():

    if len(sys.argv) < 5:
        helper = "\n Helper: python main.py train_or_test model train_data test_data\
                \n\n\t  train_or_test: training or prediction\
                \n\t    model: choose a model\
                \n\t    train_data: training dataset\
                \n\t    test_data: testing dataset"
        print(helper)
        return

    # Load training data
    data_man = dm.DataManagement(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    data_man.load_data()

    # Train model
    