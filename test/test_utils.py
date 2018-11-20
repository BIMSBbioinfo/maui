import pytest
import numpy as np
import pandas as pd

from maui import utils

def test_map_factors_to_features():
    dummy_z = pd.DataFrame(
        [
            [0,1],
            [1,0]
        ],
        index=['sample 1', 'sample 2'],
        columns=['LF1', 'LF2']
    )

    dummy_x = pd.DataFrame(
        [
            [1,1,1,0,0,0],
            [0,0,0,1,1,1]
        ],
        columns=[f'feature{i}' for i in range(6)],
        index=['sample 1', 'sample 2']
    )

    expected_corrs = np.array([
        [-1.,  1.],
        [-1.,  1.],
        [-1.,  1.],
        [ 1., -1.],
        [ 1., -1.],
        [ 1., -1.]
        ])

    corrs = utils.map_factors_to_features(dummy_z, dummy_x)

    assert np.allclose(corrs, expected_corrs)


def test_compute_roc():
    dummy_z = pd.DataFrame(
        [
            [0,1,1,1,0,1,1,0,0],
            [1,0,0,0,0,0,1,1,0],
            [1,0,1,0,0,0,1,1,0],
            [1,0,0,1,0,0,1,1,0],
            [1,0,0,0,1,1,1,1,0],
            [1,1,1,0,0,0,1,1,1],
        ],
        index=[f'sample {i}' for i in range(6)],
        columns=[f'LF{i}' for i in range(9)]
    )
    dummy_y = pd.Series(['a', 'b', 'a', 'c', 'b', 'c'], index=dummy_z.index)

    roc_curves = utils.compute_roc(dummy_z, dummy_y, cv_folds=2)
    assert np.allclose(roc_curves['a'].FPR, [0.  , 0.5 , 0.5 , 0.75, 1.  ])

def test_compute_auc():
    fpr = [0. , 0. , 0.5, 0.5, 1. ]
    tpr = [0. , 0.5, 0.5, 1. , 1. ]
    roc = utils.auc(fpr, tpr)
    assert roc - 0.75 < 1e-6

def test_estimate_km():
    yhat = pd.Series(['a','a','a','b','b','b'], index=[f'Sample {i}' for i in range(6)])
    durations = np.random.poisson(6,6)
    observed = np.random.randn(6)>.1
    survival = pd.DataFrame(dict(duration=durations, observed=observed),
        index=[f'Sample {i}' for i in range(6)])
    km = utils.estimate_kaplan_meier(yhat, survival)

    assert 'a' in km.columns
    assert 'b' in km.columns
