"""
The maui.utils model contains utility functions for multi-omics analysis
using maui.
"""

import pandas as pd
from scipy import stats

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
