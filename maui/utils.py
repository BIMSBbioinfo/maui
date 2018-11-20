"""
The maui.utils model contains utility functions for multi-omics analysis
using maui.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy import interp
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict

def map_factors_to_features(z, concatenated_data, pval_threshold=.001):
    """Compute pearson correlation of latent factors with input features.

    Parameters
    ----------
    z:                  (n_samples, n_factors) DataFrame of latent factor values, output of maui model
    concatenated_data:  (n_samples, n_features) DataFrame of concatenated multi-omics data

    Returns
    -------
    feature_s:  DataFrame (n_features, n_latent_factors)
                Latent factors representation of the data X.
    """
    feature_scores = list()
    for j in range(z.shape[1]):
        corrs = pd.DataFrame([stats.pearsonr(concatenated_data.iloc[:,i], z.iloc[:,j]) for i in range(concatenated_data.shape[1])],
                            index=concatenated_data.columns, columns=['r', 'pval'])
        corrs.loc[corrs.pval>pval_threshold, 'r'] = 0
        feature_scores.append(corrs.r)

    feature_s = pd.concat(feature_scores, axis=1)
    feature_s.columns= [i for i in range(z.shape[1])]
    return feature_s



def compute_roc(z, y, classifier=LinearSVC(C=.001), cv_folds=10):
    """Compute the ROC (false positive rate, true positive rate) using cross-validation.

    Parameters
    ----------
    z:          DataFrame (n_samples, n_latent_factors) of latent factor values
    y:          Series (n_samples,) of ground-truth labels to try to predict
    classifier: Classifier object to use, default ``LinearSVC(C=.001)``

    Returns
    -------
    roc_curves: dict, one key per class as well as "mean", each value is a dataframe
                containing the tpr (true positive rate) and fpr (falce positive rate)
                defining that class (or the mean) ROC.
    """
    class_names = sorted(y.unique())
    z_to_use = z.loc[y.index]
    y_true_bin = label_binarize(y, classes=class_names)
    y_proba = cross_val_predict(classifier, z_to_use, y, cv=cv_folds, method='decision_function')

    # Compute ROC curve and ROC area for each class
    roc_curves = dict()
    for i, cl_name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_curves[cl_name] = pd.concat([
            pd.Series(fpr, name='FPR'),
            pd.Series(tpr, name='TPR'),
        ], axis=1)

    mean_fpr = np.unique(np.concatenate([roc_curves[cl_name].FPR for cl_name in class_names]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(mean_fpr)
    for cl_name in class_names:
        mean_tpr += interp(mean_fpr, roc_curves[cl_name].FPR, roc_curves[cl_name].TPR)

    # Finally average it
    mean_tpr /= len(class_names)

    roc_curves["mean"] = pd.concat([
        pd.Series(mean_fpr, name='FPR'),
        pd.Series(mean_tpr, name='TPR'),
    ], axis=1)
    return roc_curves
