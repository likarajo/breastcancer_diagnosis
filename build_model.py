#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:00:00 2019

@author: likarajo
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
import pickle

print("Building SVM model for Breast Cancer Diagnosis...")

# ********* LOAD DATA SET ***********************************

# 1st column (col0) has the label
# 2nd (col1) to 32nd (col31) columns have the features
y_trn = np.loadtxt('data/wdbc_trn.csv', delimiter=',', usecols=(0))
X_trn = np.loadtxt('data/wdbc_trn.csv', delimiter=',', usecols=(np.arange(1,31)))
y_val = np.loadtxt('data/wdbc_val.csv', delimiter=',', usecols=(0))
X_val = np.loadtxt('data/wdbc_val.csv', delimiter=',', usecols=(np.arange(1,31)))
y_tst = np.loadtxt('data/wdbc_tst.csv', delimiter=',', usecols=(0))
X_tst = np.loadtxt('data/wdbc_tst.csv', delimiter=',', usecols=(np.arange(1,31)))

print("Data loaded.")

# ********* LEARN MODELS BY VARYING C and γ *****************

C_values = np.power(10.0, np.arange(-2.0, 3.0, 1.0))
G_values = np.power(10.0, np.arange(-2.0, 3.0, 1.0))

models = dict()
trnErr = dict()
valErr = dict()

f= open("val_trn_error.txt","w+")

f.write("{:9s}\t{:9s}\t{:9s}\t{:9s}\n".format('C', '\u03B3', 'Training Error', 'Validation Error'))
for C in C_values:
    for G in G_values:
        # Design the model (classifier) using SVC with RBF kernel
        models[(C, G)] = SVC(C=C, kernel='rbf', gamma=G)
        # Fit the model with training data
        models[(C, G)].fit(X_trn, y_trn)
        # Use Hinge Loss function for "maximum-margin" classification error
        # It gives average loss, so multiply with total no. of examples
        trnErr[(C, G)] = len(y_trn) * hinge_loss(y_trn, models[(C, G)].predict(X_trn), labels=None, sample_weight=None)
        valErr[(C, G)] = len(y_val) * hinge_loss(y_val, models[(C, G)].predict(X_val), labels=None, sample_weight=None)
        f.write("{:1f}\t{:1f}\t{:1f}\t{:1f}\n".format(C, G, trnErr[(C, G)], valErr[(C, G)]))

print("Model learning completed.")

# ********* CHOOSE BEST MODEL ******************************

# Find C and \u03B3 that produce the minimum validation error
best = min(valErr.keys(), key=(lambda k: valErr[k]))
f.write("\nMinimum validation error: {:f} at C = {:f}, \u03B3 = {:f} (best model)".format(valErr[best], best[0], best[1]))

# Finalize the model with the best C and \u03B3
model_best = SVC(C=best[0], kernel='rbf', gamma=best[1])

# Fit the finalized model with training data
model_best.fit(X_trn, y_trn)

# Find the accuracy score of the finalized model using test data
accuracy = model_best.score(X_tst, y_tst) 
f.write("\n\nAccuracy of the best SVM model is: {:f}%".format(accuracy*100))

print("Best Model selected. Accuracy {:f}%".format(accuracy*100))

f.close()

# Save the model to disk
model_file = 'SVM_model.sav'
pickle.dump(model_best, open(model_file, 'wb'))

print("Model saved --> {:s}".format(model_file))



