# Breast Cancer Diagnosis with Support Vector Machines

Used the [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) data set, which has been pre-processed and partitioned into training, validation and test sets.

**Model Learning**: Used scikit-learn's [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) function to learn SVM models with **radial-basis kernels** for each combination of $C \in \{10^{-2}, 10^{-1}, 1, 10^1, \, \cdots\, 10^4\}$ and $\gamma \in \{10^{-3}, 10^{-2}\, 10^{-1}, 1, \, 10, \, 10^2\}$, and training them with the **Training set**.

**Final Model Selection**: Used the **Validation set** to select the best classifier corresponding to the best parameter values, $C_{best}$ and $\gamma_{best}$. 

**Accuracy testing**: The accuracy of the selected model is tested on the **Test set**.
