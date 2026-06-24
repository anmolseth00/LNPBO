"""Batch-construction strategies for the GP-BO stack.

Split out of ``gp_bo.py``: KB/RKB, LP, TS, q-LogEI, GIBBON, and the mixed
discrete-continuous selector. The orchestration entry point ``select_batch``
plus ``predict`` / ``score_acquisition`` / ``_estimate_lipschitz`` remain in
``gp_bo``; this module reaches back into ``gp_bo`` for them at call time (and
for ``fit_gp``) so that monkeypatching e.g. ``gp_bo.fit_gp`` stays effective.

torch/botorch/gpytorch are optional dependencies loaded lazily by the parent
package; importing this module implies they are available.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import gpytorch
import numpy as np
import torch
from botorch.models import SingleTaskVariationalGP
from scipy.stats import norm

from ._gp_device import _to_tensor

if TYPE_CHECKING:
    from .gp_bo import GPModel


def _batch_kb(
    model: GPModel,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    acq_type: str,
    y_best: float,
    kappa: float,
    xi: float,
    randomize: bool,
) -> list[int]:
    """Kriging Believer / Randomized Kriging Believer batch selection.

    Sequentially selects points by scoring the pool, picking the best,
    then updating the model via condition_on_observations (rank-1 Cholesky
    update) with either the posterior mean (KB) or a posterior sample (RKB)
    as the fantasized observation.

    References
    ----------
    KB: Ginsbourger, D., Le Riche, R., & Carraro, L.
        "Kriging Is Well-Suited to Parallelize Optimization."
        Computational Intelligence in Expensive Optimization Problems,
        Springer, 2010, pp. 131-162.

    RKB: Sugiura, S., Takeuchi, I., & Takeno, S.
         "Randomized Kriging Believer for Parallel Bayesian Optimization
         with Regret Bounds." arXiv:2603.01470, March 2026.
    """
    from . import gp_bo

    if isinstance(model, SingleTaskVariationalGP):
        warnings.warn(
            f"{'RKB' if randomize else 'KB'} requires condition_on_observations "
            "which is not supported by variational GP. Falling back to LP.",
            stacklevel=2,
        )
        return _batch_lp(
            model,
            X_pool,
            pool_indices,
            batch_size,
            acq_type,
            y_best,
            kappa,
            xi,
        )

    device = next(model.parameters()).device
    selected = []
    available_mask = np.ones(len(X_pool), dtype=bool)
    current_model = model

    def _greedy_fill_remaining():
        """Fill remaining batch slots from the base model's posterior mean.

        Used when iterative fantasy-conditioning degenerates (NaN/non-PD
        Cholesky as fantasy points accumulate on ill-conditioned kernels).
        The base ``model`` fit fine, so a greedy posterior-mean top-up over the
        still-available pool completes the batch rather than aborting the run.
        """
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mu = model.posterior(_to_tensor(X_pool, device)).mean.squeeze(-1).cpu().numpy()
            order = np.where(available_mask)[0][np.argsort(-mu[available_mask])]
        except (RuntimeError, ValueError):
            order = np.where(available_mask)[0]
        for gi in order[: batch_size - len(selected)]:
            selected.append(int(pool_indices[int(gi)]))
            available_mask[int(gi)] = False

    for i in range(batch_size):
        try:
            scores = gp_bo.score_acquisition(
                current_model,
                X_pool[available_mask],
                acq_type,
                y_best,
                kappa,
                xi,
            )
            local_best = int(np.argmax(scores))
            global_idx = np.where(available_mask)[0][local_best]

            selected.append(int(pool_indices[global_idx]))
            available_mask[global_idx] = False

            if i < batch_size - 1:
                x_new = _to_tensor(X_pool[global_idx].reshape(1, -1), device)

                if randomize:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        posterior = current_model.posterior(x_new)
                        y_fantasy = posterior.rsample().squeeze(-1)
                else:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        y_fantasy = current_model.posterior(x_new).mean  # type: ignore[union-attr]

                cond_kwargs = {}
                likelihood = getattr(current_model, "likelihood", None)
                if isinstance(likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
                    # Fixed-noise (per-point train_Yvar) models require an explicit
                    # observation noise for the Kriging-Believer fantasy point;
                    # without it BoTorch's get_fantasy_model raises. Use the mean of
                    # the model's current per-point noise as a representative scale.
                    fantasy_noise = likelihood.noise.mean().detach()
                    cond_kwargs["noise"] = fantasy_noise.expand(y_fantasy.shape).to(y_fantasy)

                with warnings.catch_warnings():
                    # Fantasy conditioning re-triggers BoTorch input-data checks
                    # and GPyTorch numerical (jitter/Cholesky) warnings on every
                    # rank-1 update; scope the suppression to those categories.
                    from botorch.exceptions.warnings import InputDataWarning
                    from gpytorch.utils.warnings import NumericalWarning

                    warnings.filterwarnings("ignore", category=InputDataWarning)
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    current_model = current_model.condition_on_observations(
                        x_new,
                        y_fantasy,
                        **cond_kwargs,
                    )

                y_best = max(y_best, y_fantasy.item())
        except (RuntimeError, ValueError) as e:
            # Scoring or fantasy-conditioning hit a NaN / non-PD Cholesky (the
            # KB updates degenerate on ill-conditioned kernels mid-batch). Stop
            # conditioning and greedily fill the rest so the round completes
            # instead of aborting the whole run to a truncated trajectory.
            warnings.warn(
                f"KB step failed at slot {i} ({type(e).__name__}); greedy-filling "
                f"remaining {batch_size - len(selected)} of {batch_size}",
                stacklevel=2,
            )
            _greedy_fill_remaining()
            break

    return selected


def _batch_lp(
    model: GPModel,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    acq_type: str,
    y_best: float,
    kappa: float,
    xi: float,
) -> list[int]:
    """Local Penalization batch selection.

    Scores the pool once with the base acquisition, then sequentially
    selects points while applying multiplicative (or additive in log-space)
    penalties near already-selected batch points. No GP refitting needed.

    For each pending point x_j, applies soft exclusion (Proposition 1):
        phi_j(x) = Phi((L * ||x - x_j|| - r_j) / sigma(x_j))
    where L is the Lipschitz constant, r_j = y_best - mu(x_j).
    Predictions for pending points are cached incrementally.

    Reference
    ---------
    Gonzalez, J., Dai, Z., Hennig, P., & Lawrence, N.D.
    "Batch Bayesian Optimization via Local Penalization."
    AISTATS 2016. arXiv:1505.08052.
    """
    from . import gp_bo

    is_log = acq_type == "LogEI"
    selected = []
    pending_coords = []
    available_mask = np.ones(len(X_pool), dtype=bool)

    base_scores = gp_bo.score_acquisition(model, X_pool, acq_type, y_best, kappa, xi)

    # Cache LP penalty inputs: Lipschitz constant is fixed (model unchanged),
    # and pending point predictions are incrementally appended.
    L = gp_bo._estimate_lipschitz(model)
    cached_mu = []
    cached_sigma = []

    for _ in range(batch_size):
        pool_avail = X_pool[available_mask]
        scores_avail = base_scores[available_mask].copy()

        # Apply cached penalties from all pending points
        if pending_coords:
            if is_log:
                total_penalty = np.zeros(len(pool_avail))
            else:
                total_penalty = np.ones(len(pool_avail))

            for j in range(len(pending_coords)):
                r_j = max(y_best - cached_mu[j], 1e-8)
                s_j = cached_sigma[j]
                dist = np.sqrt(
                    np.sum(
                        (pool_avail - pending_coords[j].reshape(1, -1)) ** 2,
                        axis=1,
                    )
                )
                z_j = (L * dist - r_j) / s_j
                phi_j = norm.cdf(z_j)
                if is_log:
                    total_penalty += np.log(np.maximum(phi_j, 1e-10))
                else:
                    total_penalty *= phi_j

            if is_log:
                scores_avail = scores_avail + total_penalty
            else:
                scores_avail = scores_avail * total_penalty

        local_best = int(np.argmax(scores_avail))
        global_idx = np.where(available_mask)[0][local_best]

        selected.append(int(pool_indices[global_idx]))
        new_point = X_pool[global_idx].copy()
        pending_coords.append(new_point)

        # Incrementally predict only the newly added pending point
        mu_new, sigma_new = gp_bo.predict(model, new_point.reshape(1, -1))
        cached_mu.append(mu_new[0])
        cached_sigma.append(max(sigma_new[0], 1e-8))

        available_mask[global_idx] = False

    return selected


def _batch_ts(
    model: GPModel,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    seed: int,
) -> list[int]:
    """Thompson Sampling batch selection.

    For each batch slot, draws one posterior sample over the entire pool
    and selects the argmax. Each draw uses a different random seed for
    diversity. No GP refitting or penalization needed.

    Reference
    ---------
    Kandasamy, K., Krishnamurthy, A., Schneider, J., & Poczos, B.
    "Parallelised Bayesian Optimisation via Thompson Sampling."
    AISTATS 2018.
    """
    device = next(model.parameters()).device
    X_pool_t = _to_tensor(X_pool, device)

    # Draw all batch_size posterior samples in one call to avoid
    # recomputing the Cholesky-based posterior for each batch slot.
    torch.manual_seed(seed)
    # Allow extra Cholesky jitter: some kernels (e.g. Tanimoto on near-degenerate
    # fingerprint blocks with duplicate rows) yield an ill-conditioned training
    # covariance. The default double-precision jitter caps at 1e-3; raise both
    # the float and double caps so the posterior is computable. NOTE: the GP runs
    # in float64, so the `double=` value is the one that matters here.
    with (
        gpytorch.settings.cholesky_jitter(float_value=1e-2, double_value=1e-2),
        gpytorch.settings.cholesky_max_tries(6),
    ):
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = model.posterior(X_pool_t)
                # (batch_size, M, 1) -> (batch_size, M)
                all_samples = posterior.rsample(torch.Size([batch_size])).squeeze(-1)
            samples_np = all_samples.cpu().numpy()
        except (RuntimeError, ValueError) as e:
            # Still not positive-definite: fall back to mean + diagonal-variance
            # Gaussian draws so TS returns a diverse batch and the run continues
            # instead of aborting to the seed round.
            warnings.warn(
                f"TS posterior sampling failed ({type(e).__name__}); using "
                f"mean+diagonal-variance fallback for this round",
                stacklevel=2,
            )
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                post = model.posterior(X_pool_t)
                mu = post.mean.squeeze(-1).cpu().numpy()
                sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
            rng = np.random.RandomState(seed)
            samples_np = np.stack([mu + sd * rng.standard_normal(len(mu)) for _ in range(batch_size)])

    selected = []
    available_mask = np.ones(len(X_pool), dtype=bool)

    for i in range(batch_size):
        sample = samples_np[i].copy()
        sample[~available_mask] = -np.inf
        global_idx = int(np.argmax(sample))

        selected.append(int(pool_indices[global_idx]))
        available_mask[global_idx] = False

    return selected


def _batch_qlogei(
    model: GPModel,
    X_train: np.ndarray,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    device: torch.device | None = None,
) -> list[int]:
    """Joint q-batch Log Noisy Expected Improvement via BoTorch.

    Uses BoTorch's qLogNoisyExpectedImprovement with optimize_acqf_discrete
    to jointly optimize the full batch, accounting for inter-point
    correlations and observation noise. Uses X_baseline (training inputs)
    to estimate the incumbent rather than requiring a noiseless best_f.

    Reference
    ---------
    Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E.
    "Unexpected Improvements to Expected Improvement for Bayesian
    Optimization." NeurIPS 2023. arXiv:2310.20708.
    """
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim.optimize import optimize_acqf_discrete

    if device is None:
        device = next(model.parameters()).device
    X_pool_t = _to_tensor(X_pool, device)
    X_baseline_t = _to_tensor(X_train, device)

    acqf = qLogNoisyExpectedImprovement(model, X_baseline=X_baseline_t)

    with warnings.catch_warnings():
        # Discrete acqf optimization emits BoTorch input-data /
        # bad-initial-candidate warnings and GPyTorch numerical warnings;
        # scope the suppression to those rather than catching everything.
        from botorch.exceptions.warnings import (
            BadInitialCandidatesWarning,
            InputDataWarning,
        )
        from gpytorch.utils.warnings import NumericalWarning

        warnings.filterwarnings("ignore", category=InputDataWarning)
        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=NumericalWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        candidates, _ = optimize_acqf_discrete(
            acqf,
            q=batch_size,
            choices=X_pool_t,
            unique=True,
        )

    return _match_candidates_to_pool(candidates, X_pool_t, X_pool, pool_indices)


def _match_candidates_to_pool(
    candidates: torch.Tensor,
    X_pool_t: torch.Tensor,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
) -> list[int]:
    """Match optimize_acqf_discrete output back to pool indices.

    optimize_acqf_discrete returns exact rows from choices, so match
    via element-wise equality. Falls back to nearest-neighbor if needed.
    """
    selected = []
    for cand in candidates:
        match = torch.all(X_pool_t == cand.unsqueeze(0), dim=1).nonzero(as_tuple=True)[0]
        if len(match) > 0:
            selected.append(int(pool_indices[int(match[0].item())]))
        else:
            cand_np = cand.detach().cpu().numpy()
            dists = np.sum((X_pool - cand_np.reshape(1, -1)) ** 2, axis=1)
            selected.append(int(pool_indices[int(np.argmin(dists))]))
    return selected


def select_batch_mixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    pool_indices: np.ndarray | list,
    batch_size: int,
    acq_type: str,
    batch_strategy: str,
    fp_indices: list[int],
    ratio_indices: list[int],
    synth_indices: list[int],
    kappa: float = 2.576,
    xi: float = 0.01,
    seed: int = 42,
    use_sparse: bool = False,
    n_inducing: int = 512,
    kernel_type: str = "compositional",
    kernel_kwargs: dict | None = None,
    num_restarts: int = 5,
    raw_samples: int = 64,
) -> list[int]:
    """Select a batch via mixed discrete-continuous optimization.

    Enumerates unique IL identities (fingerprint feature vectors) from the
    pool, fixes them as discrete choices, and optimizes molar ratios
    continuously using BoTorch's ``optimize_acqf_mixed``. The best
    continuous candidate is matched back to the nearest pool formulation.

    Automatically degrades to standard discrete pool scoring when no ratio
    features are present (fixed-ratio studies).

    Parameters
    ----------
    X_train : (N, D) training features.
    y_train : (N,) training targets.
    X_pool : (M, D) candidate pool features.
    pool_indices : Original dataframe indices for pool rows.
    batch_size : Number of points to select.
    acq_type : Acquisition function ("UCB", "EI", "LogEI").
    batch_strategy : Batch construction ("kb" or "rkb"); other strategies
        fall back to discrete ``select_batch()``.
    fp_indices : Column indices for molecular structure features.
    ratio_indices : Column indices for compositional ratio features.
    synth_indices : Column indices for synthesis/process parameters.
    kappa : UCB exploration parameter.
    xi : EI/LogEI improvement jitter.
    seed : Random seed.
    use_sparse : Use variational GP for large training sets.
    n_inducing : Number of inducing points for sparse GP.
    kernel_type : Kernel function (default "compositional").
    kernel_kwargs : Extra kernel arguments.
    num_restarts : L-BFGS-B restarts per IL config.
    raw_samples : Raw initialization samples per restart.

    Returns
    -------
    List of selected pool_indices.
    """
    from . import gp_bo

    pool_indices = np.asarray(pool_indices)
    batch_size = min(batch_size, len(X_pool))

    # No ratio features or non-KB batch strategies → fall back to standard discrete scoring
    # Mixed optimization is only beneficial for sequential KB/RKB with ratio dims.
    # TS, qLogEI, GIBBON, LP all work directly on the discrete pool.
    if len(ratio_indices) == 0 or batch_strategy.lower() not in ("kb", "rkb"):
        return gp_bo.select_batch(
            X_train,
            y_train,
            X_pool,
            pool_indices,
            batch_size=batch_size,
            acq_type=acq_type,
            batch_strategy=batch_strategy,
            kappa=kappa,
            xi=xi,
            seed=seed,
            use_sparse=use_sparse,
            n_inducing=n_inducing,
            kernel_type=kernel_type,
            kernel_kwargs=kernel_kwargs,
        )

    if use_sparse and batch_strategy.lower() in ("kb", "rkb"):
        warnings.warn(
            "Mixed-space KB/RKB requires exact sequential conditioning. "
            "Disabling the sparse variational GP to preserve the paper-defined algorithm.",
            stacklevel=2,
        )
        use_sparse = False

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    effective_kw = kernel_kwargs or {}
    if not effective_kw.get("fp_indices"):
        effective_kw = {
            **effective_kw,
            "fp_indices": fp_indices,
            "ratio_indices": ratio_indices,
            "synth_indices": synth_indices,
        }

    model = gp_bo.fit_gp(
        X_train,
        y_train,
        use_sparse=use_sparse,
        n_inducing=n_inducing,
        _rng=rng,
        kernel_type=kernel_type,
        kernel_kwargs=effective_kw,
        seed=seed,
    )

    device = next(model.parameters()).device
    dtype = torch.float32 if device.type == "mps" else torch.float64

    # Build fixed_features_list: one entry per unique IL fingerprint vector
    non_ratio_indices = sorted(set(fp_indices) | set(synth_indices))
    X_pool_non_ratio = X_pool[:, non_ratio_indices]
    unique_configs = np.unique(X_pool_non_ratio, axis=0)

    fixed_features_list = []
    for row in unique_configs:
        ff = {}
        for col_pos, feat_idx in enumerate(non_ratio_indices):
            ff[feat_idx] = float(row[col_pos])
        fixed_features_list.append(ff)

    # Build bounds: ratio dims get [0, 1], non-ratio dims get tight bounds
    d = X_pool.shape[1]
    lower = np.full(d, 0.0)
    upper = np.full(d, 1.0)
    all_X = np.vstack([X_train, X_pool])
    for i in range(d):
        if i not in ratio_indices:
            col_min = all_X[:, i].min()
            col_max = all_X[:, i].max()
            margin = max(0.01 * (col_max - col_min), 1e-6)
            lower[i] = col_min - margin
            upper[i] = col_max + margin
        else:
            lower[i] = 1e-6  # avoid exact 0 for CLR transform
            upper[i] = 1.0
    bounds = torch.tensor(np.vstack([lower, upper]), dtype=dtype, device=device)

    # Simplex constraint on ratios: sum of ratio features = 1
    ratio_idx_t = torch.tensor(ratio_indices, dtype=torch.long, device=device)
    coeffs = torch.ones(len(ratio_indices), dtype=dtype, device=device)
    equality_constraints = [(ratio_idx_t, coeffs, 1.0)]

    # Build acquisition function
    X_train_t = _to_tensor(X_train, device)

    def _build_acqf(mdl, y_best_val):
        if acq_type == "LogEI":
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement
            return qLogNoisyExpectedImprovement(mdl, X_baseline=X_train_t)
        elif acq_type == "EI":
            from botorch.acquisition.analytic import ExpectedImprovement
            return ExpectedImprovement(mdl, best_f=y_best_val)
        else:
            from botorch.acquisition.analytic import UpperConfidenceBound
            return UpperConfidenceBound(mdl, beta=kappa**2)

    from botorch.optim.optimize import optimize_acqf_mixed

    # Sequential KB loop for batch construction
    selected = []
    available_mask = np.ones(len(X_pool), dtype=bool)
    current_model = model
    y_best_running = float(y_train.max())

    def _fallback_discrete(cur_model):
        """Score available pool discretely and select the best candidate."""
        scores = gp_bo.score_acquisition(
            cur_model, X_pool[available_mask], acq_type,
            y_best_running, kappa, xi,
        )
        local_best = int(np.argmax(scores))
        return np.where(available_mask)[0][local_best]

    def _kb_condition(cur_model, global_idx, last_slot):
        """Condition model on the selected point with KB or RKB hallucination."""
        if last_slot:
            return cur_model, None
        if isinstance(cur_model, SingleTaskVariationalGP):
            raise RuntimeError(
                "Mixed-space KB/RKB reached a variational GP after sparse mode "
                "was disabled. This path requires exact conditioning."
            )
        x_new = _to_tensor(X_pool[global_idx].reshape(1, -1), device)
        try:
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.cholesky_jitter(float_value=1e-2, double_value=1e-2),
                gpytorch.settings.cholesky_max_tries(6),
            ):
                posterior = cur_model.posterior(x_new)
                if batch_strategy.lower() == "rkb":
                    y_fantasy = posterior.rsample().squeeze(-1)
                else:
                    y_fantasy = posterior.mean  # type: ignore[union-attr]
                with warnings.catch_warnings():
                    # Mixed-space KB conditioning re-triggers BoTorch input-data
                    # checks and GPyTorch numerical (jitter) warnings.
                    from botorch.exceptions.warnings import InputDataWarning
                    from gpytorch.utils.warnings import NumericalWarning

                    warnings.filterwarnings("ignore", category=InputDataWarning)
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    conditioned = cur_model.condition_on_observations(x_new, y_fantasy)
            return conditioned, float(y_fantasy.item())
        except (RuntimeError, ValueError) as e:
            # Ill-conditioned mixed-GP posterior (NotPSD even past the default
            # 1e-3 jitter cap, which also thrashes). Skip KB conditioning for
            # this slot rather than aborting the whole run; the batch is slightly
            # less diverse but the trajectory completes.
            warnings.warn(
                f"Mixed KB conditioning failed ({type(e).__name__}); skipping "
                f"fantasy for this slot",
                stacklevel=2,
            )
            return cur_model, None

    for i in range(batch_size):
        acqf = _build_acqf(current_model, y_best_running)
        use_fallback = False

        with warnings.catch_warnings():
            # Mixed continuous-discrete acqf optimization emits BoTorch
            # input-data / bad-initial-candidate / optimization warnings and
            # GPyTorch numerical warnings; scope to those plus the generic
            # Runtime/User categories the L-BFGS-B restarts can raise.
            from botorch.exceptions.warnings import (
                BadInitialCandidatesWarning,
                InputDataWarning,
                OptimizationWarning,
            )
            from gpytorch.utils.warnings import NumericalWarning

            warnings.filterwarnings("ignore", category=InputDataWarning)
            warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            warnings.filterwarnings("ignore", category=NumericalWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                candidates, _acq_values = optimize_acqf_mixed(
                    acq_function=acqf,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features_list=fixed_features_list,
                    equality_constraints=equality_constraints,
                )
            except (RuntimeError, ValueError, torch.linalg.LinAlgError):
                use_fallback = True

        # Check simplex constraint
        if not use_fallback and ratio_indices:
            ratio_sum = candidates[0, ratio_indices].sum().item()
            if abs(ratio_sum - 1.0) > 0.01:
                use_fallback = True

        if use_fallback:
            global_idx = _fallback_discrete(current_model)
        else:
            # Match continuous candidate to nearest available pool formulation
            cand_np = candidates.squeeze(0).detach().cpu().numpy()
            pool_avail = X_pool[available_mask]
            dists = np.sum((pool_avail - cand_np.reshape(1, -1)) ** 2, axis=1)
            local_best = int(np.argmin(dists))
            global_idx = np.where(available_mask)[0][local_best]

        selected.append(int(pool_indices[global_idx]))
        available_mask[global_idx] = False

        # KB conditioning for next iteration
        current_model, fantasy_value = _kb_condition(current_model, global_idx, i == batch_size - 1)
        if fantasy_value is not None:
            y_best_running = max(y_best_running, fantasy_value)

    return selected


def _batch_gibbon(
    model: GPModel,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    device: torch.device | None = None,
    num_mv_samples: int = 10,
) -> list[int]:
    """GIBBON (General-purpose Information-Based Bayesian OptimisatioN).

    Lower bound on batch Max-value Entropy Search. Maximizes mutual
    information between observations and the optimal function value:

        alpha_GIBBON(X_batch) = (1/2) log|C| + sum_i alpha_MES(x_i)

    The log-determinant term is a DPP-like repulsion based on the
    predictive correlation matrix, encouraging batch diversity. The
    sum term rewards individual informativeness about the optimum.

    Batch selection is greedy sequential via optimize_acqf_discrete.
    When set_X_pending is called, GIBBON adds a repulsion term based
    on predictive correlations (no fantasization, no GP refitting).

    Reference
    ---------
    Moss, H.B., Leslie, D.S., Gonzalez, J., & Rayson, P.
    "GIBBON: General-purpose Information-Based Bayesian OptimisatioN."
    JMLR 22(235):1-49, 2021. arXiv:2102.03324.
    """
    from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
    from botorch.optim.optimize import optimize_acqf_discrete

    if device is None:
        device = next(model.parameters()).device
    X_pool_t = _to_tensor(X_pool, device)

    acqf = qLowerBoundMaxValueEntropy(
        model,
        candidate_set=X_pool_t,
        num_mv_samples=num_mv_samples,
        use_gumbel=True,
        maximize=True,
    )

    with warnings.catch_warnings():
        # GIBBON's discrete optimization and Gumbel max-value sampling emit
        # BoTorch input-data / bad-initial-candidate warnings and GPyTorch
        # numerical warnings; scope the suppression to those.
        from botorch.exceptions.warnings import (
            BadInitialCandidatesWarning,
            InputDataWarning,
        )
        from gpytorch.utils.warnings import NumericalWarning

        warnings.filterwarnings("ignore", category=InputDataWarning)
        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=NumericalWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        candidates, _ = optimize_acqf_discrete(
            acqf,
            q=batch_size,
            choices=X_pool_t,
            unique=True,
        )

    return _match_candidates_to_pool(candidates, X_pool_t, X_pool, pool_indices)
