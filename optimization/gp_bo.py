"""GPyTorch + BoTorch GP Bayesian Optimization with direct discrete pool scoring.

Replaces the sklearn GP + continuous optimization + nearest-neighbor pipeline.
Key speedups:
  - GPU/MPS-accelerated kernel computations via GPyTorch
  - Direct pool scoring (no continuous optimization + projection + NN matching)
  - condition_on_observations for KB/RKB (rank-1 update, not O(N^3) refit)
  - LOVE-accelerated variance via fast_pred_var
  - Optional SVGP for large N (O(NM^2) instead of O(N^3))
"""

from __future__ import annotations

import warnings

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from scipy.stats import norm

from .acquisition import _log_h_stable

# Type alias: covers SingleTaskGP, SingleTaskVariationalGP, and
# BatchedMultiOutputGPyTorchModel (returned by condition_on_observations).
type GPModel = SingleTaskGP | SingleTaskVariationalGP | Model


def get_device(use_mps: bool = False) -> torch.device:
    """Select compute device.

    MPS on Apple Silicon has limited float64 support. GPyTorch requires
    float64 for numerical stability (Cholesky, kernel evaluations).
    Default to CPU with float64; MPS uses float32 with stability caveats.
    """
    if use_mps and torch.backends.mps.is_available():
        warnings.warn(
            "MPS backend uses float32 which may cause numerical instability "
            "in GP fitting (Cholesky failures, poor hyperparameter optimization). "
            "Use CPU (default) for reliable results.",
            stacklevel=2,
        )
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor(
    X: np.ndarray, device: torch.device, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32 if device.type == "mps" else torch.float64
    return torch.tensor(np.ascontiguousarray(X), dtype=dtype, device=device)


def fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_sparse: bool = False,
    n_inducing: int = 512,
    device: torch.device | None = None,
    _rng: np.random.RandomState | None = None,
) -> SingleTaskGP | SingleTaskVariationalGP:
    """Fit a GP surrogate model.

    For N <= ~1000 (or use_sparse=False), fits an exact GP via SingleTaskGP.
    For larger N with use_sparse=True, uses SingleTaskVariationalGP with
    inducing points, reducing cost from O(N^3) to O(NM^2).

    Parameters
    ----------
    X_train : (N, D) feature matrix.
    y_train : (N,) target vector.
    use_sparse : Use variational GP with inducing points.
    n_inducing : Number of inducing points for sparse GP.
    device : Torch device. Defaults to CPU.

    Returns
    -------
    Fitted GP model in eval mode.
    """
    if device is None:
        device = get_device()

    X = _to_tensor(X_train, device)
    y = _to_tensor(y_train.ravel(), device).unsqueeze(-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if use_sparse:
            n_ind = min(n_inducing, len(X_train))
            _rng_local = _rng if _rng is not None else np.random.RandomState()
            idx = _rng_local.choice(len(X_train), n_ind, replace=False)
            inducing = X[idx].clone()
            model = SingleTaskVariationalGP(X, y, inducing_points=inducing)
            model.to(device)

            # Manual training loop for variational GP.
            # fit_gpytorch_mll uses L-BFGS-B which is inappropriate for
            # variational parameters — use Adam with VariationalELBO instead.
            mll = VariationalELBO(
                model.likelihood, model.model, num_data=len(X_train),
            )
            model.train()
            model.likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            n_epochs = min(500, max(100, len(X_train)))
            prev_loss = float("inf")
            patience, no_improve = 20, 0
            for _ in range(n_epochs):
                optimizer.zero_grad()
                output = model.model(X)
                loss = -mll(output, y.squeeze(-1))  # type: ignore[operator]
                loss.backward()
                optimizer.step()
                curr_loss = loss.item()
                if prev_loss - curr_loss < 1e-6:
                    no_improve += 1
                    if no_improve >= patience:
                        break
                else:
                    no_improve = 0
                prev_loss = curr_loss
        else:
            model = SingleTaskGP(X, y)
            model.to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

    model.eval()
    return model


def predict(
    model: GPModel,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and std over points X.

    Uses LOVE (Lanczos Variance Estimates) for fast variance computation.

    Parameters
    ----------
    model : Fitted GP model (exact or variational).
    X : (M, D) test points.

    Returns
    -------
    mean : (M,) posterior mean.
    std : (M,) posterior standard deviation.
    """
    device = next(model.parameters()).device
    X_t = _to_tensor(X, device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X_t)
        mean = posterior.mean.squeeze(-1).cpu().numpy()  # type: ignore[union-attr]
        std = posterior.variance.squeeze(-1).sqrt().cpu().numpy()  # type: ignore[union-attr]

    return mean, std


def score_acquisition(
    model: GPModel,
    X_pool: np.ndarray,
    acq_type: str,
    y_best: float,
    kappa: float = 2.576,
    xi: float = 0.01,
) -> np.ndarray:
    """Score pool points under an acquisition function.

    Parameters
    ----------
    model : Fitted GP model.
    X_pool : (M, D) candidate pool.
    acq_type : One of "UCB", "EI", "LogEI".
    y_best : Best observed value (incumbent).
    kappa : UCB exploration parameter (default: 97.5th percentile).
    xi : EI/LogEI improvement jitter.

    Returns
    -------
    scores : (M,) acquisition values (higher = more desirable).
    """
    mean, std = predict(model, X_pool)
    std = np.maximum(std, 1e-10)

    if acq_type == "UCB":
        return mean + kappa * std
    elif acq_type == "EI":
        z = (mean - y_best - xi) / std
        return std * (z * norm.cdf(z) + norm.pdf(z))
    elif acq_type == "LogEI":
        z = (mean - y_best - xi) / std
        return np.log(std) + _log_h_stable(z)
    else:
        raise ValueError(f"Unknown acq_type '{acq_type}'. Choose from: UCB, EI, LogEI")


def _estimate_lipschitz(model: GPModel) -> float:
    """Estimate Lipschitz constant L from kernel hyperparameters.

    L = sqrt(outputscale) / min(lengthscale), a conservative upper bound
    on ||d mu / dx|| for a Matern or RBF kernel.
    """
    try:
        if isinstance(model, SingleTaskVariationalGP):
            kernel = model.model.covar_module
        else:
            kernel = model.covar_module

        if hasattr(kernel, "base_kernel"):
            outputscale = kernel.outputscale.detach().cpu().item()  # type: ignore[union-attr]
            lengthscale = kernel.base_kernel.lengthscale.detach().cpu().numpy().ravel()  # type: ignore[union-attr]
            return float(np.sqrt(outputscale) / np.min(lengthscale))
        elif hasattr(kernel, "lengthscale"):
            lengthscale = kernel.lengthscale.detach().cpu().numpy().ravel()  # type: ignore[union-attr]
            return float(1.0 / np.min(lengthscale))
    except Exception:
        pass
    return 1.0


def select_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    pool_indices: np.ndarray | list,
    batch_size: int,
    acq_type: str,
    batch_strategy: str,
    kappa: float = 2.576,
    xi: float = 0.01,
    seed: int = 42,
    use_sparse: bool = False,
    n_inducing: int = 512,
) -> list[int]:
    """Select a batch of points from a discrete candidate pool.

    Parameters
    ----------
    X_train : (N, D) training features.
    y_train : (N,) training targets.
    X_pool : (M, D) candidate pool features.
    pool_indices : Original dataframe indices for pool rows.
    batch_size : Number of points to select.
    acq_type : Acquisition function ("UCB", "EI", "LogEI").
    batch_strategy : Batch construction method:
        "kb"    - Kriging Believer (Ginsbourger et al., 2010)
        "rkb"   - Randomized Kriging Believer (Sugiura et al., 2026)
        "lp"    - Local Penalization (Gonzalez et al., 2016)
        "ts"    - Thompson Sampling (Kandasamy et al., 2018)
        "qlogei" - Joint q-LogEI (Ament et al., 2023, via BoTorch)
        "gibbon" - GIBBON: information-theoretic + DPP diversity (Moss et al., 2021)
        "jes"   - Joint Entropy Search (Hvarfner et al., 2022)
    kappa : UCB exploration parameter.
    xi : EI/LogEI improvement jitter.
    seed : Random seed.
    use_sparse : Use variational GP for large training sets.
    n_inducing : Number of inducing points for sparse GP.

    Returns
    -------
    List of selected pool_indices (length batch_size or fewer if pool exhausted).
    """
    pool_indices = np.asarray(pool_indices)
    batch_size = min(batch_size, len(X_pool))

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model = fit_gp(X_train, y_train, use_sparse=use_sparse, n_inducing=n_inducing,
                   _rng=rng)
    y_best = float(y_train.max())

    strategy = batch_strategy.lower()
    device = next(model.parameters()).device
    if strategy == "qlogei":
        return _batch_qlogei(model, X_train, X_pool, pool_indices, batch_size,
                             device=device)
    elif strategy == "gibbon":
        return _batch_gibbon(model, X_pool, pool_indices, batch_size,
                             device=device)
    elif strategy == "jes":
        return _batch_jes(model, X_pool, pool_indices, batch_size, seed,
                          device=device)
    elif strategy == "ts":
        return _batch_ts(model, X_pool, pool_indices, batch_size, seed)
    elif strategy == "kb":
        return _batch_kb(
            model, X_pool, pool_indices, batch_size, acq_type,
            y_best, kappa, xi, randomize=False,
        )
    elif strategy == "rkb":
        return _batch_kb(
            model, X_pool, pool_indices, batch_size, acq_type,
            y_best, kappa, xi, randomize=True,
        )
    elif strategy == "lp":
        return _batch_lp(
            model, X_pool, pool_indices, batch_size, acq_type,
            y_best, kappa, xi,
        )
    else:
        raise ValueError(
            f"Unknown batch_strategy '{batch_strategy}'. "
            "Choose from: kb, rkb, lp, ts, qlogei, gibbon, jes"
        )


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
    if isinstance(model, SingleTaskVariationalGP):
        warnings.warn(
            f"{'RKB' if randomize else 'KB'} requires condition_on_observations "
            "which is not supported by variational GP. Falling back to LP.",
            stacklevel=2,
        )
        return _batch_lp(
            model, X_pool, pool_indices, batch_size, acq_type,
            y_best, kappa, xi,
        )

    device = next(model.parameters()).device
    selected = []
    available_mask = np.ones(len(X_pool), dtype=bool)
    current_model = model

    for i in range(batch_size):
        scores = score_acquisition(
            current_model, X_pool[available_mask], acq_type, y_best, kappa, xi,
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

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                current_model = current_model.condition_on_observations(
                    x_new, y_fantasy,
                )

            y_best = max(y_best, y_fantasy.item())

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
    is_log = acq_type == "LogEI"
    selected = []
    pending_coords = []
    available_mask = np.ones(len(X_pool), dtype=bool)

    base_scores = score_acquisition(model, X_pool, acq_type, y_best, kappa, xi)

    # Cache LP penalty inputs: Lipschitz constant is fixed (model unchanged),
    # and pending point predictions are incrementally appended.
    L = _estimate_lipschitz(model)
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
                dist = np.sqrt(np.sum(
                    (pool_avail - pending_coords[j].reshape(1, -1)) ** 2, axis=1,
                ))
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
        mu_new, sigma_new = predict(model, new_point.reshape(1, -1))
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
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X_pool_t)
        # (batch_size, M, 1) -> (batch_size, M)
        all_samples = posterior.rsample(
            torch.Size([batch_size]),
        ).squeeze(-1)

    selected = []
    available_mask = np.ones(len(X_pool), dtype=bool)
    samples_np = all_samples.cpu().numpy()

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
        warnings.filterwarnings("ignore")
        candidates, _ = optimize_acqf_discrete(
            acqf, q=batch_size, choices=X_pool_t, unique=True,
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
        warnings.filterwarnings("ignore")
        candidates, _ = optimize_acqf_discrete(
            acqf, q=batch_size, choices=X_pool_t, unique=True,
        )

    return _match_candidates_to_pool(candidates, X_pool_t, X_pool, pool_indices)


def _batch_jes(
    model: GPModel,
    X_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    seed: int = 42,
    device: torch.device | None = None,
    num_optima: int = 64,
) -> list[int]:
    """Joint Entropy Search batch selection.

    Maximizes mutual information between observations and the joint
    optimal location-value pair (x*, f*). Tighter bound than GIBBON
    since it conditions on both where and what the optimum is.

    Optimal input-output samples are generated via Thompson sampling
    on the discrete pool: draw posterior samples and take argmax per
    sample. This avoids the continuous get_optimal_samples utility
    which is inappropriate for discrete pool settings.

    Reference
    ---------
    Hvarfner, C., Hutter, F., & Nardi, L.
    "Joint Entropy Search for Maximally-Informed Bayesian Optimization."
    NeurIPS 2022. arXiv:2206.04771.
    """
    from botorch.acquisition.joint_entropy_search import qJointEntropySearch
    from botorch.optim.optimize import optimize_acqf_discrete

    if device is None:
        device = next(model.parameters()).device
    X_pool_t = _to_tensor(X_pool, device)

    # Generate optimal input-output samples via Thompson sampling
    torch.manual_seed(seed)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X_pool_t)
        samples = posterior.rsample(torch.Size([num_optima]))  # (num_optima, M, 1)
        best_idx = samples.squeeze(-1).argmax(dim=-1)          # (num_optima,)
        optimal_inputs = X_pool_t[best_idx]                    # (num_optima, D)
        optimal_outputs = samples[                             # (num_optima, 1)
            torch.arange(num_optima), best_idx, :
        ]

    acqf = qJointEntropySearch(
        model,
        optimal_inputs=optimal_inputs,
        optimal_outputs=optimal_outputs,
        condition_noiseless=True,
        estimation_type="LB",
        num_samples=min(num_optima, 32),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        candidates, _ = optimize_acqf_discrete(
            acqf, q=batch_size, choices=X_pool_t, unique=True,
        )

    return _match_candidates_to_pool(candidates, X_pool_t, X_pool, pool_indices)
