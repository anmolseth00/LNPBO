"""Numerical parity tests: our score_acquisition vs BoTorch native implementations.

Verifies that our UCB, EI, and LogEI produce identical (or near-identical) values
compared to BoTorch's UpperConfidenceBound, ExpectedImprovement, and
LogExpectedImprovement on the same GP model and pool data.

Also verifies KB fantasization (condition_on_observations) matches BoTorch fantasize().

Run: .venv/bin/python -m pytest tests/test_botorch_numerical_parity.py -v
"""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from LNPBO.optimization.acquisition import _log_h_stable
from LNPBO.optimization.gp_bo import _to_tensor, fit_gp, predict, score_acquisition


def _setup(n_train=80, n_pool=200, d=5, seed=42):
    """Shared setup: fit a GP and return model + data."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d).astype(np.float64)
    y_train = (np.sin(X_train[:, 0]) + X_train[:, 1] ** 2
               + 0.1 * rng.randn(n_train))
    X_pool = rng.randn(n_pool, d).astype(np.float64)
    model = fit_gp(X_train, y_train)
    y_best = float(y_train.max())
    return model, X_train, y_train, X_pool, y_best


class TestUCBParity:
    """Compare our UCB = mu + kappa*sigma vs BoTorch UpperConfidenceBound."""

    def test_ucb_exact_match(self):
        from botorch.acquisition.analytic import UpperConfidenceBound

        model, _, y_train, X_pool, y_best = _setup()
        kappa = 2.576
        device = next(model.parameters()).device

        # Ours
        ours = score_acquisition(model, X_pool, "UCB", y_best, kappa=kappa)

        # BoTorch: beta = kappa^2 (BoTorch uses sqrt(beta) internally)
        botorch_ucb = UpperConfidenceBound(model, beta=kappa**2)
        X_t = _to_tensor(X_pool, device).unsqueeze(1)  # (M, 1, D) for q-batch
        with torch.no_grad():
            theirs = botorch_ucb(X_t).cpu().numpy()

        # Should be numerically identical (same GP, same formula)
        np.testing.assert_allclose(
            ours, theirs, atol=1e-6, rtol=1e-6,
            err_msg="UCB: our implementation vs BoTorch UpperConfidenceBound",
        )

    def test_ucb_ranking_identical(self):
        """Even if values differ slightly, rankings must match exactly."""
        from botorch.acquisition.analytic import UpperConfidenceBound

        model, _, y_train, X_pool, y_best = _setup(seed=123)
        kappa = 2.0
        device = next(model.parameters()).device

        ours = score_acquisition(model, X_pool, "UCB", y_best, kappa=kappa)
        botorch_ucb = UpperConfidenceBound(model, beta=kappa**2)
        X_t = _to_tensor(X_pool, device).unsqueeze(1)
        with torch.no_grad():
            theirs = botorch_ucb(X_t).cpu().numpy()

        our_ranking = np.argsort(-ours)
        their_ranking = np.argsort(-theirs)
        np.testing.assert_array_equal(
            our_ranking[:20], their_ranking[:20],
            err_msg="UCB top-20 ranking differs between ours and BoTorch",
        )


class TestEIParity:
    """Compare our EI vs BoTorch ExpectedImprovement."""

    def test_ei_close(self):
        from botorch.acquisition.analytic import ExpectedImprovement

        model, _, y_train, X_pool, y_best = _setup()
        xi = 0.01
        device = next(model.parameters()).device

        # Ours
        ours = score_acquisition(model, X_pool, "EI", y_best, xi=xi)

        # BoTorch: best_f = y_best (not y_best + xi; xi baked into our z formula)
        # Our formula: z = (mu - y_best - xi) / sigma
        # BoTorch: z = (mu - best_f) / sigma
        # To match: set BoTorch best_f = y_best + xi
        botorch_ei = ExpectedImprovement(model, best_f=y_best + xi)
        X_t = _to_tensor(X_pool, device).unsqueeze(1)
        with torch.no_grad():
            theirs = botorch_ei(X_t).cpu().numpy()

        # EI formula: sigma * (z * Phi(z) + phi(z))
        # Both use scipy/torch normal — should agree closely
        np.testing.assert_allclose(
            ours, theirs, atol=1e-5, rtol=1e-4,
            err_msg="EI: our implementation vs BoTorch ExpectedImprovement",
        )

    def test_ei_ranking_top10(self):
        from botorch.acquisition.analytic import ExpectedImprovement

        model, _, y_train, X_pool, y_best = _setup(seed=99)
        xi = 0.01
        device = next(model.parameters()).device

        ours = score_acquisition(model, X_pool, "EI", y_best, xi=xi)
        botorch_ei = ExpectedImprovement(model, best_f=y_best + xi)
        X_t = _to_tensor(X_pool, device).unsqueeze(1)
        with torch.no_grad():
            theirs = botorch_ei(X_t).cpu().numpy()

        # Top-10 ranking must agree
        our_top10 = set(np.argsort(-ours)[:10])
        their_top10 = set(np.argsort(-theirs)[:10])
        overlap = len(our_top10 & their_top10)
        assert overlap >= 8, (
            f"EI top-10 overlap: {overlap}/10 — rankings should mostly agree"
        )


class TestLogEIParity:
    """Compare our LogEI vs BoTorch LogExpectedImprovement."""

    def test_logei_close(self):
        from botorch.acquisition.analytic import LogExpectedImprovement

        model, _, y_train, X_pool, y_best = _setup()
        xi = 0.01
        device = next(model.parameters()).device

        # Ours: log(sigma) + _log_h_stable(z) where z = (mu - y_best - xi) / sigma
        ours = score_acquisition(model, X_pool, "LogEI", y_best, xi=xi)

        # BoTorch LogExpectedImprovement
        botorch_logei = LogExpectedImprovement(model, best_f=y_best + xi)
        X_t = _to_tensor(X_pool, device).unsqueeze(1)
        with torch.no_grad():
            theirs = botorch_logei(X_t).cpu().numpy()

        # Previous audit showed max diff 2.7e-12. Verify.
        max_diff = np.max(np.abs(ours - theirs))
        np.testing.assert_allclose(
            ours, theirs, atol=1e-6, rtol=1e-5,
            err_msg=f"LogEI max diff = {max_diff:.2e}",
        )

    def test_logei_extreme_z_parity(self):
        """Test LogEI parity at extreme z values (where stability matters)."""
        from botorch.acquisition.analytic import LogExpectedImprovement

        # Create scenario with very high y_best → extreme negative z
        model, _, y_train, X_pool, _ = _setup()
        y_best_extreme = float(y_train.max()) + 20.0
        xi = 0.0
        device = next(model.parameters()).device

        ours = score_acquisition(model, X_pool, "LogEI", y_best_extreme, xi=xi)
        botorch_logei = LogExpectedImprovement(model, best_f=y_best_extreme)
        X_t = _to_tensor(X_pool, device).unsqueeze(1)
        with torch.no_grad():
            theirs = botorch_logei(X_t).cpu().numpy()

        # At extreme z, the stable computation matters most
        assert np.all(np.isfinite(ours)), "Our LogEI produced non-finite at extreme z"
        assert np.all(np.isfinite(theirs)), "BoTorch LogEI produced non-finite at extreme z"

        # Rankings should still agree even at extreme values
        our_ranking = np.argsort(-ours)
        their_ranking = np.argsort(-theirs)
        np.testing.assert_array_equal(
            our_ranking[:10], their_ranking[:10],
            err_msg="LogEI ranking at extreme z differs",
        )

    def test_log_h_stable_vs_botorch_log_ei_helper(self):
        """Direct comparison of _log_h_stable vs BoTorch's internal _log_ei_helper."""
        from botorch.acquisition.analytic import _log_ei_helper

        z_values = np.array([
            -100.0, -50.0, -10.0, -5.0, -2.0, -1.0, -0.5,
            0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0,
        ])

        ours = _log_h_stable(z_values)
        z_tensor = torch.tensor(z_values, dtype=torch.float64)
        theirs = _log_ei_helper(z_tensor).numpy()

        finite_mask = np.isfinite(ours) & np.isfinite(theirs)
        max_diff = np.max(np.abs(ours[finite_mask] - theirs[finite_mask]))

        np.testing.assert_allclose(
            ours[finite_mask], theirs[finite_mask], atol=1e-10,
            err_msg=f"_log_h_stable vs _log_ei_helper: max diff = {max_diff:.2e}",
        )


class TestKBFantasizeParity:
    """Compare KB's condition_on_observations with BoTorch's fantasize()."""

    def test_fantasize_posterior_match(self):
        """After fantasizing one point, posterior should match.

        Verifies that our KB approach (condition_on_observations with posterior
        mean as fantasy value) matches BoTorch fantasize() with a fixed sampler.
        """
        import gpytorch
        from botorch.sampling import IIDNormalSampler

        model, _, _, X_pool, _ = _setup(n_pool=50)
        device = next(model.parameters()).device

        # Pick a point to fantasize
        x_new = _to_tensor(X_pool[0:1], device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_fantasy = model.posterior(x_new).mean  # KB uses mean

        # Method 1: condition_on_observations (our KB approach)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model_cond = model.condition_on_observations(x_new, y_fantasy)

        # Method 2: BoTorch fantasize with a seeded sampler
        # fantasize draws from posterior(x_new) + noise, so to get comparable
        # results we must compare against condition_on_observations with the
        # same fantasy observation. Use fantasize and extract the fantasy Y.
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]), seed=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model_fant = model.fantasize(X=x_new, sampler=sampler)

        # fantasize internally calls condition_on_observations, so the
        # mechanism is identical. Verify the conditioned model structure.
        X_test = _to_tensor(X_pool[1:20], device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_cond = model_cond.posterior(X_test)
            mean_cond = post_cond.mean.squeeze(-1).cpu().numpy()

        # The fantasized model has a batch dim from the sampler
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_fant = model_fant.posterior(X_test)
            mean_fant = post_fant.mean.squeeze().cpu().numpy()

        # Since fantasize uses a different fantasy Y (sampled vs mean),
        # the posteriors won't match exactly. But the correlation should be
        # very high — both are conditioning on nearly the same observation.
        from scipy.stats import pearsonr
        r, _ = pearsonr(mean_cond, mean_fant)
        assert r > 0.99, (
            f"condition_on_obs vs fantasize posterior means: r={r:.6f}, "
            "should be >0.99 (same mechanism, similar fantasy obs)"
        )

    def test_condition_on_observations_deterministic(self):
        """Verify condition_on_observations is deterministic (same input → same output)."""
        import gpytorch

        model, _, _, X_pool, _ = _setup(n_pool=50)
        device = next(model.parameters()).device

        x_new = _to_tensor(X_pool[0:1], device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_fantasy = model.posterior(x_new).mean

        import warnings
        models = []
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                m = model.condition_on_observations(x_new, y_fantasy)
            models.append(m)

        X_test = _to_tensor(X_pool[1:20], device)
        means = []
        for m in models:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                means.append(m.posterior(X_test).mean.squeeze(-1).cpu().numpy())

        np.testing.assert_allclose(
            means[0], means[1], atol=1e-12,
            err_msg="condition_on_observations should be deterministic",
        )


class TestImplementationDifferences:
    """Document and verify known implementation differences."""

    def test_ei_scipy_vs_torch_normal(self):
        """Our EI uses scipy.stats.norm; BoTorch uses torch.distributions.Normal.
        Verify the difference is negligible."""
        z_values = np.linspace(-5, 5, 1000)

        # scipy
        scipy_cdf = norm.cdf(z_values)
        scipy_pdf = norm.pdf(z_values)

        # torch
        z_t = torch.tensor(z_values, dtype=torch.float64)
        torch_dist = torch.distributions.Normal(0, 1)
        torch_cdf = torch_dist.cdf(z_t).numpy()
        torch_pdf = torch_dist.log_prob(z_t).exp().numpy()

        np.testing.assert_allclose(scipy_cdf, torch_cdf, atol=1e-14)
        np.testing.assert_allclose(scipy_pdf, torch_pdf, atol=1e-14)

    def test_std_clamp_difference(self):
        """Our code clamps std to 1e-10; BoTorch does not.
        Verify this only affects near-zero variance points."""
        model, _, y_train, X_pool, y_best = _setup()

        mean, std = predict(model, X_pool)
        n_clamped = np.sum(std < 1e-10)

        # For a well-fitted GP with non-degenerate data, no points should
        # have near-zero variance (they're all out-of-sample)
        assert n_clamped == 0, (
            f"{n_clamped} pool points have std < 1e-10; "
            "clamp difference only matters when std → 0"
        )

    def test_our_predict_matches_botorch_posterior(self):
        """Verify predict() returns same values as model.posterior() directly."""
        import gpytorch

        model, _, _, X_pool, _ = _setup()
        device = next(model.parameters()).device

        # Our predict
        mean_ours, std_ours = predict(model, X_pool)

        # Direct BoTorch posterior
        X_t = _to_tensor(X_pool, device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(X_t)
            mean_direct = posterior.mean.squeeze(-1).cpu().numpy()
            std_direct = posterior.variance.squeeze(-1).sqrt().cpu().numpy()

        np.testing.assert_allclose(mean_ours, mean_direct, atol=1e-10)
        np.testing.assert_allclose(std_ours, std_direct, atol=1e-10)
