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

    fpr, tpr = utils.compute_roc(dummy_z, dummy_y, cv_folds=2)
    assert np.allclose(fpr['a'], [0.  , 0.5 , 0.5 , 0.75, 1.  ])
