"""Tests for optimization.serialization checkpoint save/load."""

import json

import numpy as np
import pytest

from LNPBO.optimization.serialization import load_checkpoint, save_checkpoint


@pytest.fixture
def tmp_ckpt(tmp_path):
    return tmp_path / "ckpt"


class TestSaveLoadSklearnGP:
    def test_round_trip(self, tmp_ckpt):
        from sklearn.gaussian_process import GaussianProcessRegressor

        X = np.random.default_rng(42).standard_normal((20, 3))
        y = X[:, 0] + 0.1 * np.random.default_rng(42).standard_normal(20)
        gp = GaussianProcessRegressor().fit(X, y)

        save_checkpoint(
            tmp_ckpt,
            model=gp,
            surrogate_type="gp_sklearn",
            feature_columns=["f1", "f2", "f3"],
            round_number=3,
        )

        model, meta, scaler = load_checkpoint(tmp_ckpt)
        assert meta["surrogate_type"] == "gp_sklearn"
        assert meta["feature_columns"] == ["f1", "f2", "f3"]
        assert meta["round_number"] == 3
        assert scaler is None
        # Verify predictions match
        pred_orig = gp.predict(X[:5])
        pred_loaded = model.predict(X[:5])
        np.testing.assert_allclose(pred_orig, pred_loaded)


class TestSaveLoadXGBoost:
    def test_round_trip(self, tmp_ckpt):
        pytest.importorskip("xgboost")
        from xgboost import XGBRegressor

        X = np.random.default_rng(42).standard_normal((30, 4))
        y = X[:, 0] * 2 + X[:, 1]
        xgb = XGBRegressor(n_estimators=10, verbosity=0).fit(X, y)

        save_checkpoint(
            tmp_ckpt,
            model=xgb,
            surrogate_type="xgb",
            feature_columns=["a", "b", "c", "d"],
        )

        model, meta, scaler = load_checkpoint(tmp_ckpt)
        assert meta["surrogate_type"] == "xgb"
        pred_orig = xgb.predict(X[:5])
        pred_loaded = model.predict(X[:5])
        np.testing.assert_allclose(pred_orig, pred_loaded)


class TestCheckpointMetadata:
    def test_fields_present(self, tmp_ckpt):
        from sklearn.linear_model import Ridge

        model = Ridge().fit(np.eye(3), [1, 2, 3])
        save_checkpoint(
            tmp_ckpt,
            model=model,
            surrogate_type="ridge",
            feature_columns=["x1", "x2", "x3"],
            round_number=5,
            extra_metadata={"study_id": "test123"},
        )

        meta_path = tmp_ckpt / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "version" in meta
        assert "timestamp" in meta
        assert meta["surrogate_type"] == "ridge"
        assert meta["round_number"] == 5
        assert meta["study_id"] == "test123"

    def test_with_scaler(self, tmp_ckpt):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X = np.random.default_rng(42).standard_normal((10, 2))
        scaler = StandardScaler().fit(X)
        model = Ridge().fit(scaler.transform(X), [1] * 10)

        save_checkpoint(
            tmp_ckpt,
            model=model,
            surrogate_type="ridge",
            feature_columns=["a", "b"],
            scaler=scaler,
        )

        _, _, loaded_scaler = load_checkpoint(tmp_ckpt)
        assert loaded_scaler is not None
        np.testing.assert_allclose(scaler.mean_, loaded_scaler.mean_)


class TestCheckpointEdgeCases:
    def test_load_nonexistent_dir(self, tmp_ckpt):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_ckpt / "does_not_exist")

    def test_overwrite_checkpoint(self, tmp_ckpt):
        from sklearn.linear_model import Ridge

        model_v1 = Ridge().fit(np.eye(3), [1, 2, 3])
        model_v2 = Ridge().fit(np.eye(3), [4, 5, 6])

        save_checkpoint(tmp_ckpt, model=model_v1, surrogate_type="ridge",
                        feature_columns=["x"], round_number=1)
        save_checkpoint(tmp_ckpt, model=model_v2, surrogate_type="ridge",
                        feature_columns=["x"], round_number=2)

        loaded, meta, _ = load_checkpoint(tmp_ckpt)
        assert meta["round_number"] == 2
        np.testing.assert_allclose(
            loaded.predict(np.eye(3)), model_v2.predict(np.eye(3))
        )
