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
from sklearn.preprocessing import StandardScaler
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

def estimate_kaplan_meier(y, survival,
    duration_column='duration', observed_column='observed'):
    """Estimate survival curves for groups defined in y based on survival data in ``survival``

    Parameters
    ----------
    y:                  pd.Series, groups (clusters, subtypes). the index is
                        the sample names
    survival:           pd.DataFrame with the same index as y, with columns for
                        the duration (survival time for each patient) and whether
                        or not the death was observed. If the death was not
                        observed (sensored), the duration is the time of the last
                        followup.
    duration_column:        the name of the column in  ``survival`` with the duration
    observed_column:    the name of the column in ``survival`` with True/False values
                        for whether death was observed or not

    Returns
    -------
    km_estimates:       pd.DataFrame, index is the timeline, columns are survival
                        functions (estimated by Kaplan-Meier) for each class, as
                        defined in ``y``.
    """
    try:
        import lifelines
    except ImportError:
        raise ImportError('The module ``lifelines`` was not found. It is required for this functionality. You may install it using `pip install lifelines`.')
    kmf = lifelines.KaplanMeierFitter()
    sfs = dict()
    for cl in y.unique():
        ixs = list(set(y[y==cl].index) & set(survival.index))
        kmf.fit(survival.loc[ixs][duration_column],
            survival.loc[ixs][observed_column], label=cl)
        sfs[cl] = kmf.survival_function_
    return pd.concat([sfs[k] for k in sorted(y.unique())], axis=1).interpolate()

def multivariate_logrank_test(y, survival,
    duration_column='duration', observed_column='observed'):
    """Compute the multivariate log-rank test for differential survival
    among the groups defined by ``y`` in the survival data in ``survival``,
    under the null-hypothesis that all groups have the same survival function
    (i.e. test whether at least one group has different survival rates)

    Parameters
    ----------
    y:                  pd.Series, groups (clusters, subtypes). the index is
                        the sample names
    survival:           pd.DataFrame with the same index as y, with columns for
                        the duration (survival time for each patient) and whether
                        or not the death was observed. If the death was not
                        observed (sensored), the duration is the time of the last
                        followup.
    duration_column:        the name of the column in  ``survival`` with the duration
    observed_column:    the name of the column in ``survival`` with True/False values
                        for whether death was observed or not

    Returns
    -------
    test_statistic:     the test statistic (chi-square)
    p_value:            the associated p_value
    """
    try:
        import lifelines
    except ImportError:
        raise ImportError('The module ``lifelines`` was not found. It is required for this functionality. You may install it using `pip install lifelines`.')
    ixs = list(set(y.index) & set(survival.index))
    mlr = lifelines.statistics.multivariate_logrank_test(survival.loc[ixs][duration_column],
                                                         y.loc[ixs],
                                                         survival.loc[ixs][observed_column])
    return mlr.test_statistic, mlr.p_value

def select_clinical_factors(z, survival,
    duration_column='duration', observed_column='observed', alpha=.05,
    cox_penalizer=0):
    """Select latent factors which are predictive of survival. This is
    accomplished by fitting a Cox Proportional Hazards (CPH) model to each
    latent factor, while controlling for known covariates, and only keeping
    those latent factors whose coefficient in the CPH is nonzero (adjusted
    p-value < alpha).

    Parameters
    ----------
    survival:           pd.DataFrame of survival information and relevant covariates
                        (such as sex, age at diagnosis, or tumor stage)
    duration_column:    the name of the column in ``survival`` containing the
                        duration (time between diagnosis and death or last followup)
    observed_column:    the name of the column in ``survival`` containing
                        indicating whether time of death is known
    alpha:              threshold for p-value of CPH coefficients to call a latent
                        factor clinically relevant (p < alpha)
    cox_penalizer:      penalty coefficient in Cox PH solver (see ``lifelines.CoxPHFitter``)

    Returns
    -------
    z_clinical: pd.DataFrame, subset of the latent factors which have been
                determined to have clinical value (are individually predictive
                of survival, controlling for covariates)
    """
    cox_coefficients = _cph_coefs(z, survival,
        duration_column, observed_column, penalizer=cox_penalizer)
    signif_cox_coefs = cox_coefficients.T[cox_coefficients.T.p<alpha]
    return z.loc[:,signif_cox_coefs.index]

def _cph_coefs(z, survival, duration_column, observed_column, penalizer=0):
    """Compute one CPH model for each latent factor (column) in z.
    Return summaries (beta values, p values, confidence intervals)
    """
    try:
        import lifelines
    except ImportError:
        raise ImportError('The module ``lifelines`` was not found. It is required for this functionality. You may install it using `pip install lifelines`.')
    return pd.concat([
        lifelines.CoxPHFitter(penalizer=penalizer).fit(survival.assign(LF=z.loc[:,i]).dropna(),
            duration_column, observed_column).summary.loc['LF'].rename(i)
        for i in z.columns], axis=1)

def compute_harrells_c(z, survival, duration_column='duration',
    observed_column='observed', cox_penalties=[.1,1,10,100,1000,10000],
    cv_folds=5):
    """Compute's Harrell's c-Index for a Cox Proportional Hazards regression modeling
    survival by the latent factors in z.

    Parameters
    ----------
    z:                  pd.DataFrame (n_samples, n_latent factors)
    survival:           pd.DataFrame of survival information and relevant covariates
                        (such as sex, age at diagnosis, or tumor stage)
    duration_column:    the name of the column in ``survival`` containing the
                        duration (time between diagnosis and death or last followup)
    observed_column:    the name of the column in ``survival`` containing
                        indicating whether time of death is known
    cox_penalties:      penalty coefficient in Cox PH solver (see ``lifelines.CoxPHFitter``)
                        to try. Returns the best c given by the different penalties
                        (by cross-validation)
    cv_folds:           number of cross-validation folds to compute C

    Returns
    -------
    cs: array, Harrell's c-Index, an auc-like metric for survival prediction accuracy.
        one value per cv_fold

    """
    cvcs = [_cv_coxph_c(
        z,
        survival,
        p,
        duration_column,
        observed_column,
        cv_folds) for p in cox_penalties]
    return cvcs[np.argmax([np.median(e) for e in cvcs])]

def _cv_coxph_c(z, survival, penalty,
    duration_column='duration', observed_column='observed', cv_folds=5):
    try:
        import lifelines
        import lifelines.utils
    except ImportError:
        raise ImportError('The module ``lifelines`` was not found. It is required for this functionality. You may install it using `pip install lifelines`.')

    cph = lifelines.CoxPHFitter(penalizer=penalty)
    survdf = pd.concat([survival, z], axis=1, sort=False).dropna()

    scores = lifelines.utils.k_fold_cross_validation(cph,
        survdf, duration_column, event_col=observed_column, k=cv_folds)
    return scores

def scale(df):
    """Scale and center data

    Parameters
    ----------
    df:     pd.DataFrame (n_features, n_samples) non-scaled data

    Returns
    -------
    scaled: pd.DataFrame (n_features, n_samples) scaled data
    """
    df_scaled = StandardScaler().fit_transform(df.T)
    return pd.DataFrame(df_scaled,
        columns=df.index,
        index=df.columns).T
