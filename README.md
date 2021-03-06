# Breast Cancer Diagnosis with Support Vector Machines

Used the [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) Diagnostic data set, which has been pre-processed and partitioned into training, validation and test sets.

**Model Learning**: Used scikit-learn's [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) function to learn SVM models with **radial-basis-function (RBF) kernels** for each combination of ***C*** and ***γ*** in {10<sup>-2</sup>, 10<sup>-1</sup>, 1, 10<sup>1</sup>, 10<sup>2</sup>}, and training them with the *Training set*.

**Loss computation**: Used scikit-learn's [metrics.**hinge_loss**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html) function to find the *maximum-margin classification error*, and compute the training and validation error for each combination of tuning parameters (C and γ).

**Final Model Selection**: Used the *Validation set* to select the best classifier corresponding to the best parameter values, C<sub>best</sub> and γ<sub>best</sub>, for which the *validation error is minimum*.
Then fit the finalized model with the training data.

**Accuracy testing**: The accuracy ***score*** of the selected model is computed on the *Test set*.

---

## Running

### Building the model
```
$ python3 build_model.py
Building SVM model for Breast Cancer Diagnosis...
Data loaded.
Model learning completed.
Best Model selected. Accuracy 96.521739%
Model saved --> SVM_model.sav
```

### Predicting with the model
```
$ python3 predict.py SVM_model.sav
[1.]
```
