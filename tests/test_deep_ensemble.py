"""Minimal smoke tests for models.deep_ensemble.DeepEnsemble."""

import numpy as np
import pytest

from LNPBO.models.deep_ensemble import DeepEnsemble


@pytest.fixture()
def random_data():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 5).astype(np.float32)
    y = rng.randn(50).astype(np.float32)
    return X, y


def test_instantiation():
    ens = DeepEnsemble(input_dim=5, n_models=3, epochs=2)
    assert ens.n_models == 3
    assert ens.input_dim == 5


def test_fit(random_data):
    X, y = random_data
    ens = DeepEnsemble(input_dim=5, n_models=3, epochs=5)
    ens.fit(X, y)
    assert len(ens.models) == 3


def test_predict_shapes(random_data):
    X, y = random_data
    ens = DeepEnsemble(input_dim=5, n_models=3, epochs=5)
    ens.fit(X, y)
    mu, sigma = ens.predict(X)
    assert mu.shape == (50,)
    assert sigma.shape == (50,)


def test_sigma_nonnegative(random_data):
    X, y = random_data
    ens = DeepEnsemble(input_dim=5, n_models=3, epochs=5)
    ens.fit(X, y)
    _, sigma = ens.predict(X)
    assert np.all(sigma >= 0)


def test_sigma_positive_somewhere(random_data):
    """With n_models=3 and different seeds, ensemble members should disagree."""
    X, y = random_data
    ens = DeepEnsemble(input_dim=5, n_models=3, epochs=10)
    ens.fit(X, y)
    _, sigma = ens.predict(X)
    assert np.any(sigma > 0), "Expected nonzero uncertainty from ensemble disagreement"


def test_hidden_dims_are_used(random_data):
    X, y = random_data
    ens = DeepEnsemble(input_dim=5, n_models=1, hidden_dims=(32, 16), epochs=2)
    ens.fit(X, y, bootstrap=False)
    model = ens.models[0]

    assert model.hidden[0].out_features == 32
    assert model.hidden[1].out_features == 16
