import pytest
import numpy as np
import pandas as pd

from maui import Maui


samples = [f'Sample_{i}' for i in range(10)]

n_features_1 = 20
df1 = pd.DataFrame(np.random.randn(n_features_1, len(samples)),
    columns=samples,
    index=[f'data1_feature_{i}' for i in range(n_features_1)])
n_features_2 = 6
df2 = pd.DataFrame(np.random.randn(n_features_2, len(samples)),
    columns=samples,
    index=[f'data2_feature_{i}' for i in range(n_features_2)])

df_empty = pd.DataFrame(np.random.randn(0, len(samples)),
    columns=samples,
    index=[f'data0_feature_{i}' for i in range(0)])

def test_validate_X_fails_if_not_dict():
    maui_model = Maui()
    with pytest.raises(Exception):
        maui_model._validate_X([1,2,3])

def test_validate_X_fails_if_samples_mismatch():
    maui_model = Maui()
    with pytest.raises(Exception):
        df2_bad = df2.iloc[:,:2]
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

def test_dict2array():
    maui_model = Maui()
    arr = maui_model._dict2array({'data1': df1, 'data2': df2})
    assert arr.shape[0] == len(df1.columns)
    assert arr.shape[1] == len(df1.index) + len(df2.index)

def test_maui_saves_feature_correlations():
    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    z = maui_model.fit_transform({'d1': df1, 'd2': df2})
    assert hasattr(maui_model, 'feature_correlations')
