#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:12:40 2019

@author: likarajo
"""
import sys
import pickle

# ********** LOAD THE MODEL **************************

model_file = sys.argv[1]
classifier = pickle.load(open(model_file, 'rb'))

# ********** GET INPUT DATA **************************

X = [[1.8563, -0.40903, 1.7368, 1.9608, -0.70896, -0.35361, 0.11824, 0.67307, 0.20199, -0.82776, 0.63048, -0.90663, 0.38631, 1.0174, -0.46504, -0.62871, -0.39826, 0.4693, -0.76653, 0.073478, 1.7664, -0.48712, 1.5042, 1.905, -0.34307, -0.35661, -0.057546, 1.151, -0.18772, 0.38733]]

# ********** PREDICT USING MODEL *********************

label = classifier.predict(X)

print(label)