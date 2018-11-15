import pytest
import numpy as np
import pandas as pd

from maui import Maui


samples = [f'Sample_{i}' for i in range(10)]

n_features_1 = 20
df1 = pd.DataFrame(np.random.randn(len(samples), n_features_1),
    index=samples,
    columns=[f'data1_feature_{i}' for i in range(n_features_1)])
n_features_2 = 6
df2 = pd.DataFrame(np.random.randn(len(samples), n_features_2),
    index=samples,
    columns=[f'data2_feature_{i}' for i in range(n_features_2)])

df_empty = pd.DataFrame(np.random.randn(len(samples), 0),
    index=samples,
    columns=[f'data0_feature_{i}' for i in range(0)])

def test_validate_X_fails_if_not_dict():
    maui_model = Maui()
    with pytest.raises(Exception):
        maui_model._validate_X([1,2,3])

def test_validate_X_fails_if_samples_mismatch():
    maui_model = Maui()
    with pytest.raises(Exception):
        df2_bad = df2.iloc[:2,:]
        data_with_mismatching_samples = {'a': df1, 'b': df2_bad}
        maui_model._validate_X(data_with_mismatching_samples)

def test_validate_X_fails_if_some_data_empty():
    maui_model = Maui()
    with pytest.raises(Exception):
        maui_model._validate_X({'a': df1, 'e': df_empty})

def test_validate_X_returns_true_on_valid_data():
    maui_model = Maui()
    valid_data = {'a': df1, 'b': df2}
    assert maui_model._validate_X(valid_data)   