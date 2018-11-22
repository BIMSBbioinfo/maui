import pytest
from unittest import mock

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
    with pytest.raises(ValueError):
        maui_model._validate_X([1,2,3])

def test_validate_X_fails_if_samples_mismatch():
    maui_model = Maui()
    with pytest.raises(ValueError):
        df2_bad = df2.iloc[:,:2]
        data_with_mismatching_samples = {'a': df1, 'b': df2_bad}
        maui_model._validate_X(data_with_mismatching_samples)

def test_validate_X_fails_if_some_data_empty():
    maui_model = Maui()
    with pytest.raises(ValueError):
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

def test_maui_clusters_with_single_k():
    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    maui_model.z_ = pd.DataFrame(np.random.randn(10,2),
        index=[f'sample {i}' for i in range(10)],
        columns=['LF1', 'LF2'])
    maui_model.x_ = pd.DataFrame(np.random.randn(20,10),
        index=[f'feature {i}' for i in range(20)],
        columns=[f'sample {i}' for i in range(10)])

    yhat = maui_model.cluster(5)
    assert yhat.shape == (10,)

def test_maui_clusters_picks_optimal_k_by_ami():
    ami_mock = mock.Mock()
    ami_mock.side_effect = [2,3,1] # the optimal AMI will be given at the second trial
    with mock.patch('sklearn.metrics.adjusted_mutual_info_score', ami_mock):
        maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
        maui_model.z_ = pd.DataFrame(np.random.randn(10,2),
            index=[f'sample {i}' for i in range(10)],
            columns=['LF1', 'LF2'])
        maui_model.x_ = pd.DataFrame(np.random.randn(20,10),
            index=[f'feature {i}' for i in range(20)],
            columns=[f'sample {i}' for i in range(10)])

        the_y = pd.Series(np.arange(10), index=maui_model.z_.index)

        maui_model.cluster(ami_y=the_y, optimal_k_range=[1,2,3]) # the second trial is k=2

        assert maui_model.optimal_k_ == 2

def test_maui_clusters_picks_optimal_k_by_silhouette():
    silhouette_mock = mock.Mock()
    silhouette_mock.side_effect = [2,3,1] # the optimal silhouette will be given at the second trial
    with mock.patch('sklearn.metrics.silhouette_score', silhouette_mock):
        maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
        maui_model.z_ = pd.DataFrame(np.random.randn(10,2),
            index=[f'sample {i}' for i in range(10)],
            columns=['LF1', 'LF2'])
        maui_model.x_ = pd.DataFrame(np.random.randn(20,10),
            index=[f'feature {i}' for i in range(20)],
            columns=[f'sample {i}' for i in range(10)])
        maui_model.cluster(optimal_k_method='silhouette', optimal_k_range=[1,2,3]) # the second trial is k=2

        assert maui_model.optimal_k_ == 2

def test_maui_clusters_picks_optimal_k_with_custom_scoring():
    scorer = mock.Mock()
    scorer.side_effect = [2,3,1] # the optimal AMI will be given at the second trial
    scorer.__name__ = 'mock_scorer'

    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    maui_model.z_ = pd.DataFrame(np.random.randn(10,2),
        index=[f'sample {i}' for i in range(10)],
        columns=['LF1', 'LF2'])
    maui_model.x_ = pd.DataFrame(np.random.randn(20,10),
        index=[f'feature {i}' for i in range(20)],
        columns=[f'sample {i}' for i in range(10)])
    maui_model.cluster(optimal_k_method=scorer, optimal_k_range=[1,2,3]) # the second trial is k=2

    assert maui_model.optimal_k_ == 2

def test_maui_computes_roc_and_auc():
    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    maui_model.z_ = pd.DataFrame(
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
    y = pd.Series(['a', 'b', 'a', 'c', 'b', 'c'], index=maui_model.z_.index)
    rocs = maui_model.compute_roc(y, cv_folds=2)
    assert rocs == maui_model.roc_curves_
    assert 'a' in rocs
    assert 'b' in rocs
    assert 'c' in rocs
    assert "mean" in rocs

    aucs = maui_model.compute_auc(y, cv_folds=2)
    assert aucs == maui_model.aucs_

def test_maui_clusters_only_samples_in_y_index_when_optimizing():
    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    maui_model.z_ = pd.DataFrame(np.random.randn(10,2),
        index=[f'sample {i}' for i in range(10)],
        columns=['LF1', 'LF2'])
    maui_model.x_ = pd.DataFrame(np.random.randn(20,10),
        index=[f'feature {i}' for i in range(20)],
        columns=[f'sample {i}' for i in range(10)])

    y = pd.Series(['a','a','a','b','b','b'],
        index=[f'sample {i}' for i in range(6)])

    yhat = maui_model.cluster(ami_y=y, optimal_k_range=[1,2,3])
    assert set(yhat.index) == set(y.index)

def test_select_clinical_factors():
    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    maui_model.z_ = pd.DataFrame(
        [
            [1,1,1,0,0,0,1,0,1],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,1,0,0,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,1,1,1,1],
        ],
        index=[f'sample {i}' for i in range(11)],
        columns=[f'LF{i}' for i in range(9)]
    ) # here the first 3 factors separate the groups and the last 6 do not

    durations = [1,2,3,4,5,6, 1000,2000,3000, 4000, 5000] # here the first 3 have short durations, the last 3 longer ones
    observed = [True]*11 # all events observed
    survival = pd.DataFrame(dict(duration=durations, observed=observed),
        index=[f'sample {i}' for i in range(11)])

    z_clin = maui_model.select_clinical_factors(survival, cox_penalizer=1)
    assert 'LF0' in z_clin.columns
    assert 'LF5' not in z_clin.columns

def test_maui_computes_harrells_c():
    maui_model = Maui(n_hidden=[10], n_latent=2, epochs=1)
    maui_model.z_ = pd.DataFrame(
        [
            [1,1,1,0,0,0,1,0,1],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,0,1,1,1,0],
            [1,1,1,1,1,0,0,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,0,1,1,0],
            [0,0,0,1,0,1,1,1,1],
        ],
        index=[f'sample {i}' for i in range(11)],
        columns=[f'LF{i}' for i in range(9)]
    ) # here the first 3 factors separate the groups and the last 6 do not

    durations = [1,2,3,4,5,6, 1000,2000,3000, 4000, 5000] # here the first 3 have short durations, the last 3 longer ones
    observed = [True]*11 # all events observed
    survival = pd.DataFrame(dict(duration=durations, observed=observed),
        index=[f'sample {i}' for i in range(11)])
    cs = maui_model.c_index(survival, clinical_only=True,
        duration_column='duration', observed_column='observed',
        cox_penalties=[.1,1,10,100,1000,10000],
        cv_folds=2, sel_clin_alpha=.05, sel_clin_penalty=1)
    print(cs)
    assert np.allclose(cs, [.8,.8])
