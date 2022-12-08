# -*- coding: utf-8 -*-

#####
# Céline ZHANG (zhac3201)
# Omar CHIDA (chim2708)
###

import numpy as np
import sys
from tools.DataLoader import DataLoader


dl = DataLoader("data/train.csv", class_col_name="species", excluded_features={"id"})
dl.load()
for label, features in dl.get_dataset():
    print(label)
    print(features)
    print(dl.get_label_name(label))
    break