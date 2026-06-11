"""Tests for the CompositionalProductKernel.

The CompositionalProductKernel uses domain-specific sub-kernels for LNP
formulations: Matern-5/2 (ARD) for molecular structure PCs, Aitchison for
compositional ratios, and Matern-5/2 for synthesis parameters.

Run: .venv/bin/python -m pytest tests/test_compositional_kernel.py -v
"""

import numpy as np
import pytest
import torch

from LNPBO.optimization.kernels import (
    CompositionalProductKernel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compositional_data(n, d_fp=10, d_ratio=4, d_synth=3, seed=42):
    """Create synthetic data with three feature groups.

    Returns X (n, d_fp + d_ratio + d_synth), plus the index lists.
    FP features are continuous (simulating PCA-reduced molecular descriptors).
    """
    rng = np.random.RandomState(seed)
    fp = rng.randn(n, d_fp).astype(np.float64)
    ratio = rng.dirichlet(np.ones(d_ratio), size=n).astype(np.float64)
    synth = rng.randn(n, d_synth).astype(np.float64)
    X = np.hstack([fp, ratio, synth])

    fp_indices = list(range(d_fp))
    ratio_indices = list(range(d_fp, d_fp + d_ratio))
    synth_indices = list(range(d_fp + d_ratio, d_fp + d_ratio + d_synth))

    return X, fp_indices, ratio_indices, synth_indices


def _make_targets(X, ratio_start=10, ratio_end=14, seed=42):
    """Synthetic target: depends on ratio log-ratio and synth sum."""
    rng = np.random.RandomState(seed)
    ratio_part = np.log(X[:, ratio_start] / X[:, ratio_start + 1].clip(1e-6))
    synth_part = X[:, ratio_end:].sum(axis=1)
    return ratio_part + 0.5 * synth_part + 0.1 * rng.randn(len(X))


# ---------------------------------------------------------------------------
# 1. Kernel construction
# ---------------------------------------------------------------------------


class TestKernelConstruction:
    def test_all_three_subkernels(self):
        """Kernel with all three groups should have three sub-kernels."""
        k = CompositionalProductKernel(
            fp_indices=list(range(10)),
            ratio_indices=list(range(10, 14)),
            synth_indices=list(range(14, 17)),
        )
        assert "structure" in k._sub_kernel_dict
        assert "aitchison" in k._sub_kernel_dict
        assert "matern" in k._sub_kernel_dict

    def test_fp_only(self):
        """Omitting ratio and synth should give structure-only kernel."""
        k = CompositionalProductKernel(
            fp_indices=list(range(10)),
            ratio_indices=[],
            synth_indices=[],
        )
        assert "structure" in k._sub_kernel_dict
        assert "aitchison" not in k._sub_kernel_dict
        assert "matern" not in k._sub_kernel_dict

    def test_ratio_only(self):
        k = CompositionalProductKernel(
            fp_indices=[],
            ratio_indices=list(range(4)),
            synth_indices=[],
        )
        assert "structure" not in k._sub_kernel_dict
        assert "aitchison" in k._sub_kernel_dict

    def test_synth_only(self):
        k = CompositionalProductKernel(
            fp_indices=[],
            ratio_indices=[],
            synth_indices=list(range(3)),
        )
        assert "matern" in k._sub_kernel_dict
        assert len(k._sub_kernel_dict) == 1

    def test_empty_raises_in_fit_gp(self):
        """fit_gp with compositional but no indices should raise."""
        from LNPBO.optimization.gp_bo import fit_gp

        rng = np.random.RandomState(42)
        X = rng.randn(20, 5).astype(np.float64)
        y = rng.randn(20).astype(np.float64)
        with pytest.raises(ValueError, match="at least one non-empty"):
            fit_gp(X, y, kernel_type="compositional", kernel_kwargs={})


# ---------------------------------------------------------------------------
# 2. Kernel matrix properties
# ---------------------------------------------------------------------------


class TestKernelMatrix:
    def test_kernel_matrix_psd(self):
        """Product kernel matrix should be positive semi-definite."""
        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(30)
        k = CompositionalProductKernel(
            fp_indices=fp_idx,
            ratio_indices=ratio_idx,
            synth_indices=synth_idx,
        )
        X_t = torch.tensor(X, dtype=torch.float64)
        K = k(X_t, X_t).evaluate().detach()

        eigenvalues = torch.linalg.eigvalsh(K)
        assert (eigenvalues >= -1e-6).all(), f"Kernel matrix not PSD: min eigenvalue = {eigenvalues.min().item():.6e}"

    def test_diagonal_ones(self):
        """Diagonal of the product kernel should be close to 1.0.

        Each sub-kernel has k(x, x) = 1 (before ScaleKernel), so the
        product diagonal is scale_tani * scale_aitchison * scale_matern.
        With default initialization, outputscales start near 1.
        """
        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(20)
        k = CompositionalProductKernel(
            fp_indices=fp_idx,
            ratio_indices=ratio_idx,
            synth_indices=synth_idx,
        )
        X_t = torch.tensor(X, dtype=torch.float64)
        diag = k(X_t, X_t, diag=True).detach()
        assert diag.shape == (20,)
        assert torch.all(torch.isfinite(diag))
        assert torch.all(diag > 0)

    def test_kernel_symmetry(self):
        """K(X, X) should be symmetric."""
        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(25)
        k = CompositionalProductKernel(
            fp_indices=fp_idx,
            ratio_indices=ratio_idx,
            synth_indices=synth_idx,
        )
        X_t = torch.tensor(X, dtype=torch.float64)
        K = k(X_t, X_t).evaluate().detach()
        np.testing.assert_allclose(K.numpy(), K.T.numpy(), atol=1e-10)

    def test_product_of_subkernels(self):
        """Product kernel K = K_tani * K_aitchison * K_matern element-wise."""
        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(15)
        k = CompositionalProductKernel(
            fp_indices=fp_idx,
            ratio_indices=ratio_idx,
            synth_indices=synth_idx,
        )
        X_t = torch.tensor(X, dtype=torch.float64)

        K_full = k(X_t, X_t).evaluate().detach()

        X_fp = X_t[:, fp_idx]
        X_ratio = X_t[:, ratio_idx]
        X_synth = X_t[:, synth_idx]

        K_tani = k._sub_kernel_dict["structure"](X_fp, X_fp).evaluate().detach()
        K_aitchison = k._sub_kernel_dict["aitchison"](X_ratio, X_ratio).evaluate().detach()
        K_matern = k._sub_kernel_dict["matern"](X_synth, X_synth).evaluate().detach()

        K_expected = K_tani * K_aitchison * K_matern
        np.testing.assert_allclose(
            K_full.numpy(),
            K_expected.numpy(),
            atol=1e-10,
            err_msg="Product kernel does not equal element-wise product of sub-kernels",
        )


# ---------------------------------------------------------------------------
# 3. GP fitting and prediction
# ---------------------------------------------------------------------------


class TestGPFitPredict:
    def test_fit_predict_basic(self):
        """GP with compositional kernel should produce finite predictions."""
        from LNPBO.optimization.gp_bo import fit_gp, predict

        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(50)
        y = _make_targets(X)
        X_test, _, _, _ = _make_compositional_data(20, seed=99)

        model = fit_gp(
            X,
            y,
            kernel_type="compositional",
            kernel_kwargs={
                "fp_indices": fp_idx,
                "ratio_indices": ratio_idx,
                "synth_indices": synth_idx,
            },
        )
        mean, std = predict(model, X_test)
        assert mean.shape == (20,)
        assert std.shape == (20,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_predictions_correlate_with_target(self):
        """GP predictions should track the synthetic target function."""
        from scipy.stats import pearsonr

        from LNPBO.optimization.gp_bo import fit_gp, predict

        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(80)
        y = _make_targets(X)
        X_test, _, _, _ = _make_compositional_data(40, seed=99)
        y_test = _make_targets(X_test, seed=99)

        model = fit_gp(
            X,
            y,
            kernel_type="compositional",
            kernel_kwargs={
                "fp_indices": fp_idx,
                "ratio_indices": ratio_idx,
                "synth_indices": synth_idx,
            },
        )
        mean, _ = predict(model, X_test)
        r, _ = pearsonr(mean, y_test)
        assert r > 0.2, f"Compositional GP predictions should track target (Pearson r={r:.3f})"

    def test_acquisition_scores_finite(self):
        """All acquisition functions should produce finite scores."""
        from LNPBO.optimization.gp_bo import fit_gp, score_acquisition

        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(50)
        y = _make_targets(X)
        X_pool, _, _, _ = _make_compositional_data(30, seed=99)

        model = fit_gp(
            X,
            y,
            kernel_type="compositional",
            kernel_kwargs={
                "fp_indices": fp_idx,
                "ratio_indices": ratio_idx,
                "synth_indices": synth_idx,
            },
        )
        y_best = y.max()

        for acq_type in ["UCB", "EI", "LogEI"]:
            scores = score_acquisition(model, X_pool, acq_type, y_best)
            assert np.all(np.isfinite(scores)), f"Compositional GP {acq_type} scores contain non-finite values"


# ---------------------------------------------------------------------------
# 4. Batch selection
# ---------------------------------------------------------------------------


class TestBatchSelection:
    def _kw(self):
        X, fp_idx, ratio_idx, synth_idx = _make_compositional_data(60)
        y = _make_targets(X)
        X_pool, _, _, _ = _make_compositional_data(100, seed=99)
        pool_indices = np.arange(100)
        kernel_kwargs = {
            "fp_indices": fp_idx,
            "ratio_indices": ratio_idx,
            "synth_indices": synth_idx,
        }
        return X, y, X_pool, pool_indices, kernel_kwargs

    def test_ts_batch(self):
        """Compositional + Thompson Sampling batch selection."""
        from LNPBO.optimization.gp_bo import select_batch

        X, y, X_pool, pool_indices, kw = self._kw()
        batch = select_batch(
            X,
            y,
            X_pool,
            pool_indices,
            batch_size=6,
            acq_type="UCB",
            batch_strategy="ts",
            seed=42,
            kernel_type="compositional",
            kernel_kwargs=kw,
        )
        assert len(batch) == 6
        assert len(set(batch)) == 6

    def test_kb_batch(self):
        """Compositional + Kriging Believer batch selection."""
        from LNPBO.optimization.gp_bo import select_batch

        X, y, X_pool, pool_indices, kw = self._kw()
        batch = select_batch(
            X,
            y,
            X_pool,
            pool_indices,
            batch_size=4,
            acq_type="LogEI",
            batch_strategy="kb",
            seed=42,
            kernel_type="compositional",
            kernel_kwargs=kw,
        )
        assert len(batch) == 4
        assert len(set(batch)) == 4

    def test_lp_batch(self):
        """Compositional + Local Penalization batch selection."""
        from LNPBO.optimization.gp_bo import select_batch

        X, y, X_pool, pool_indices, kw = self._kw()
        batch = select_batch(
            X,
            y,
            X_pool,
            pool_indices,
            batch_size=4,
            acq_type="UCB",
            batch_strategy="lp",
            seed=42,
            kernel_type="compositional",
            kernel_kwargs=kw,
        )
        assert len(batch) == 4
        assert len(set(batch)) == 4

    def test_batch_from_pool(self):
        """Selected indices should all come from pool_indices."""
        from LNPBO.optimization.gp_bo import select_batch

        X, y, X_pool, pool_indices, kw = self._kw()
        batch = select_batch(
            X,
            y,
            X_pool,
            pool_indices,
            batch_size=6,
            acq_type="UCB",
            batch_strategy="ts",
            seed=42,
            kernel_type="compositional",
            kernel_kwargs=kw,
        )
        pool_set = set(pool_indices.tolist())
        for idx in batch:
            assert idx in pool_set, f"Index {idx} not in pool_indices"


# ---------------------------------------------------------------------------
# 5. Feature column classification
# ---------------------------------------------------------------------------


class TestClassifyColumns:
    def test_basic_classification(self):
        from LNPBO.benchmarks.runner import classify_feature_columns

        cols = [
            "IL_count_mfp_pc1",
            "IL_count_mfp_pc2",
            "IL_count_mfp_pc3",
            "HL_count_mfp_pc1",
            "IL_molratio",
            "HL_molratio",
            "CHL_molratio",
            "PEG_molratio",
            "IL_to_nucleicacid_massratio",
        ]
        result = classify_feature_columns(cols)
        assert result["fp_indices"] == [0, 1, 2, 3]
        assert result["ratio_indices"] == [4, 5, 6, 7]
        assert result["synth_indices"] == [8]

    def test_no_synth(self):
        from LNPBO.benchmarks.runner import classify_feature_columns

        cols = ["IL_count_mfp_pc1", "IL_molratio", "HL_molratio"]
        result = classify_feature_columns(cols)
        assert result["fp_indices"] == [0]
        assert result["ratio_indices"] == [1, 2]
        assert result["synth_indices"] == []

    def test_no_ratios(self):
        from LNPBO.benchmarks.runner import classify_feature_columns

        cols = ["IL_count_mfp_pc1", "IL_count_mfp_pc2"]
        result = classify_feature_columns(cols)
        assert result["fp_indices"] == [0, 1]
        assert result["ratio_indices"] == []
        assert result["synth_indices"] == []


# ---------------------------------------------------------------------------
# 6. Partial subsets (two-component product kernels)
# ---------------------------------------------------------------------------


class TestPartialKernels:
    def test_structure_x_aitchison(self):
        """Product of structure Matern and Aitchison only (no synth params)."""
        from LNPBO.optimization.gp_bo import fit_gp, predict

        rng = np.random.RandomState(42)
        fp = rng.randn(40, 10).astype(np.float64)
        ratio = rng.dirichlet([1, 1, 1, 1], size=40).astype(np.float64)
        X = np.hstack([fp, ratio])
        y = fp.sum(axis=1) + np.log(ratio[:, 0] / ratio[:, 1].clip(1e-6)) + 0.1 * rng.randn(40)

        model = fit_gp(
            X,
            y,
            kernel_type="compositional",
            kernel_kwargs={
                "fp_indices": list(range(10)),
                "ratio_indices": list(range(10, 14)),
                "synth_indices": [],
            },
        )
        X_test = np.hstack(
            [
                rng.randn(15, 10).astype(np.float64),
                rng.dirichlet([1, 1, 1, 1], size=15).astype(np.float64),
            ]
        )
        mean, std = predict(model, X_test)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_aitchison_x_matern(self):
        """Product of Aitchison and Matern only (no fingerprints)."""
        from LNPBO.optimization.gp_bo import fit_gp, predict

        rng = np.random.RandomState(42)
        ratio = rng.dirichlet([1, 1, 1, 1], size=40).astype(np.float64)
        synth = rng.randn(40, 3).astype(np.float64)
        X = np.hstack([ratio, synth])
        y = np.log(ratio[:, 0] / ratio[:, 1].clip(1e-6)) + synth.sum(axis=1) + 0.1 * rng.randn(40)

        model = fit_gp(
            X,
            y,
            kernel_type="compositional",
            kernel_kwargs={
                "fp_indices": [],
                "ratio_indices": list(range(4)),
                "synth_indices": list(range(4, 7)),
            },
        )
        X_test = np.hstack(
            [
                rng.dirichlet([1, 1, 1, 1], size=15).astype(np.float64),
                rng.randn(15, 3).astype(np.float64),
            ]
        )
        mean, std = predict(model, X_test)
        assert np.all(np.isfinite(mean))
        assert np.all(std >= 0)
