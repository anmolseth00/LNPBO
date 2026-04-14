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

import os
import warnings
from typing import TYPE_CHECKING

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from scipy.stats import norm

from .acquisition import _log_h_stable

KERNEL_TYPES = {"matern", "tanimoto", "aitchison", "dkl", "rf", "compositional", "robust"}

if TYPE_CHECKING:
    from typing import Union

    # Type alias: covers SingleTaskGP, SingleTaskVariationalGP, and
    # BatchedMultiOutputGPyTorchModel (returned by condition_on_observations).
    GPModel = Union[SingleTaskGP, SingleTaskVariationalGP, Model]


def get_device(use_mps: bool | None = None) -> torch.device:
    """Select compute device: CUDA > CPU > MPS (opt-in).

    Priority:
      1. CUDA if available — full float64 support, best performance.
      2. CPU (default) — reliable float64, always works.
      3. MPS (opt-in) — Apple Silicon GPU, float32 only, may cause
         numerical instability in GP fitting.

    MPS opt-in resolution order: explicit ``use_mps`` arg, else the
    ``LNPBO_USE_MPS`` env var (``1``/``true``/``yes``), else False.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if use_mps is None:
        use_mps = os.environ.get("LNPBO_USE_MPS", "").lower() in {"1", "true", "yes"}
    if use_mps and torch.backends.mps.is_available():
        warnings.warn(
            "MPS backend uses float32 which may cause numerical instability "
            "in GP fitting (Cholesky failures, poor hyperparameter optimization). "
            "Use CPU or CUDA for reliable results.",
            stacklevel=2,
        )
        return torch.device("mps")
    return torch.device("cpu")


def _to_tensor(
    X: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert a NumPy array to a contiguous torch Tensor on the given device.

    Args:
        X: Input array.
        device: Target torch device (CPU or MPS).
        dtype: Torch dtype. Defaults to float32 on MPS, float64 on CPU.

    Returns:
        Contiguous Tensor on the specified device with the chosen dtype.
    """
    if dtype is None:
        dtype = torch.float32 if device.type == "mps" else torch.float64
    return torch.tensor(np.ascontiguousarray(X), dtype=dtype, device=device)


class _DKLFeatureExtractor(torch.nn.Module):
    """MLP feature extractor for Deep Kernel Learning.

    Maps D-dimensional inputs to a lower-dimensional learned representation
    through a 3-layer MLP with ReLU activations.

    Reference
    ---------
    Wilson, A.G., Hu, Z., Salakhutdinov, R., & Xing, E.P.
    "Deep Kernel Learning." AISTATS 2016. arXiv:1511.02222.
    """

    def __init__(self, input_dim: int, output_dim: int = 16):
        """Initialize a 3-layer MLP (input_dim -> 64 -> 32 -> output_dim).

        Args:
            input_dim: Dimensionality of the raw input features.
            output_dim: Dimensionality of the learned representation.
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP feature extractor.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Learned representation of shape (..., output_dim).
        """
        return self.net(x)


class _DKLInputTransform(torch.nn.Module):
    """BoTorch-compatible InputTransform wrapping a frozen feature extractor.

    Implements the minimal InputTransform interface so that SingleTaskGP
    applies the DKL feature mapping transparently in posterior(),
    condition_on_observations(), and all downstream BoTorch utilities.

    Reference
    ---------
    Wilson, A.G., Hu, Z., Salakhutdinov, R., & Xing, E.P.
    "Deep Kernel Learning." AISTATS 2016. arXiv:1511.02222.
    """

    is_one_to_many: bool = False
    transform_on_train: bool = True
    transform_on_eval: bool = True
    transform_on_fantasize: bool = True

    def __init__(self, feature_extractor: _DKLFeatureExtractor):
        """Wrap a frozen feature extractor as a BoTorch InputTransform.

        Args:
            feature_extractor: Trained _DKLFeatureExtractor whose parameters
                will be frozen (requires_grad=False).
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the frozen feature extractor to input X.

        Args:
            X: Input tensor of shape (..., input_dim).

        Returns:
            Transformed tensor of shape (..., output_dim).
        """
        return self.feature_extractor(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the frozen feature extractor (BoTorch InputTransform API).

        Args:
            X: Input tensor of shape (..., input_dim).

        Returns:
            Transformed tensor of shape (..., output_dim).
        """
        return self.feature_extractor(X)

    def preprocess_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Preprocess transform (delegates to transform).

        Args:
            X: Input tensor.

        Returns:
            Transformed tensor.
        """
        return self.transform(X)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse transform (not supported for DKL).

        Raises:
            NotImplementedError: DKL feature extraction is not invertible.
        """
        raise NotImplementedError("DKL transform is not invertible")

    def equals(self, other) -> bool:
        """Check if another transform is of the same type.

        Args:
            other: Another InputTransform instance.

        Returns:
            True if other is a _DKLInputTransform.
        """
        return isinstance(other, _DKLInputTransform)


def _fit_dkl_gp(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    n_epochs: int = 200,
    lr: float = 0.01,
    output_dim: int = 16,
) -> SingleTaskGP:
    """Jointly train a DKL model (feature extractor + GP) with Adam.

    Phase 1: Build a temporary SingleTaskGP with the feature extractor as
    input_transform, then optimize the joint marginal log-likelihood over
    all parameters (NN weights + GP hyperparameters) using Adam.

    Phase 2: Freeze the feature extractor, transform training inputs once,
    and construct a fresh SingleTaskGP on the transformed features. This
    gives a clean model where condition_on_observations, posterior, etc.
    all work via BoTorch's standard input_transform machinery.

    Reference
    ---------
    Wilson, A.G., Hu, Z., Salakhutdinov, R., & Xing, E.P.
    "Deep Kernel Learning." AISTATS 2016. arXiv:1511.02222.
    """
    input_dim = X.shape[-1]
    feature_extractor = _DKLFeatureExtractor(input_dim, output_dim).to(device)
    dtype = X.dtype
    for p in feature_extractor.parameters():
        p.data = p.data.to(dtype)

    # Phase 1: joint training with a temporary model
    # Build GP on transformed features for initialization
    with torch.no_grad():
        X_init = feature_extractor(X)
    temp_model = SingleTaskGP(X_init, y)
    temp_model.to(device)

    temp_model.train()
    temp_model.likelihood.train()
    mll = ExactMarginalLogLikelihood(temp_model.likelihood, temp_model)

    all_params = list(feature_extractor.parameters()) + list(temp_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    prev_loss = float("inf")
    patience, no_improve = 20, 0
    for _ in range(n_epochs):
        optimizer.zero_grad()
        X_feat = feature_extractor(X)
        # Manually set the training data to the current transformed inputs
        # so the GP's forward pass uses updated features each step.
        temp_model.set_train_data(X_feat, y.squeeze(-1), strict=False)
        output = temp_model(X_feat)
        loss = -mll(output, y.squeeze(-1))
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

    # Phase 2: freeze NN and build final model with input_transform
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    input_transform = _DKLInputTransform(feature_extractor)
    model = SingleTaskGP(X, y, input_transform=input_transform)
    model.to(device)

    # Copy learned GP hyperparameters from the jointly trained temp model
    # (the fresh SingleTaskGP has default hyperparameters)
    with torch.no_grad():
        for name, param in temp_model.named_parameters():
            target = dict(model.named_parameters()).get(name)
            if target is not None and target.shape == param.shape:
                target.copy_(param)

    import contextlib

    mll_final = ExactMarginalLogLikelihood(model.likelihood, model)
    # Short L-BFGS polish on GP hyperparameters only (NN is frozen)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with contextlib.suppress(RuntimeError, ValueError):
            fit_gpytorch_mll(mll_final)

    model.eval()
    return model


def _fit_rf_kernel_gp(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    n_estimators: int = 500,
    rf_seed: int = 42,
) -> SingleTaskGP:
    """Fit an exact GP with a Random Forest proximity kernel.

    The RF kernel K(x_i, x_j) = fraction of trees where x_i and x_j
    land in the same leaf. This is data-adaptive and PSD (Scornet 2016).
    The RF is fit once on (X, y); the resulting kernel matrix is used
    as the GP's covariance function.

    Because the RF kernel has no differentiable hyperparameters, GP
    hyperparameter optimization is limited to the noise variance and
    output scale (via the wrapping ScaleKernel). L-BFGS fitting may
    converge quickly or fail silently — both are acceptable since the
    kernel's expressiveness comes from the RF, not from GP hyperparams.

    Reference
    ---------
    Scornet, E. (2016). "Random Forests and Kernel Methods."
        IEEE Transactions on Information Theory, 62(3), 1485-1500.
    """
    from gpytorch.kernels import ScaleKernel

    from .rf_kernel import RandomForestKernel

    rf_kernel = RandomForestKernel(
        X,
        y,
        n_estimators=n_estimators,
        random_state=rf_seed,
    )
    covar_module = ScaleKernel(rf_kernel)

    model = SingleTaskGP(X, y, covar_module=covar_module)
    model.to(device)

    import contextlib

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with contextlib.suppress(RuntimeError, ValueError):
            fit_gpytorch_mll(mll)

    model.eval()
    return model


def _fit_robust_gp(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> SingleTaskGP:
    """Fit a Robust GP via BoTorch's RobustRelevancePursuitSingleTaskGP.

    Automatically identifies and downweights unreliable observations via
    Bayesian model selection (Relevance Pursuit), without manual outlier
    removal.

    Reference
    ---------
    Ament, S. et al. (2024). "Robust Gaussian Processes via Relevance Pursuit."
    arXiv:2410.24222.
    """
    from botorch.models.robust_relevance_pursuit_model import (
        RobustRelevancePursuitSingleTaskGP,
    )

    model = RobustRelevancePursuitSingleTaskGP(
        train_X=X,
        train_Y=y,
        cache_model_trace=True,
    )
    model.to(device)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        import contextlib
        with contextlib.suppress(RuntimeError, ValueError):
            fit_gpytorch_mll(
                mll,
                fractions_of_outliers=[0.0, 0.05, 0.1, 0.2],
                timeout_sec=300.0,
            )

    model.eval()
    return model


def fit_multitask_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_indices: np.ndarray,
    device: torch.device | None = None,
    rank: int = 3,
) -> "Model":
    """Fit a Multi-Task GP with low-rank ICM coregionalization.

    Each task/study shares a base kernel but has a learned inter-task
    covariance, allowing the GP to transfer information across studies.

    Parameters
    ----------
    X_train : (N, D) feature matrix.
    y_train : (N,) targets.
    task_indices : (N,) integer task/study IDs.
    device : Torch device. Defaults to CPU.
    rank : Rank of the ICM task covariance matrix.

    Returns
    -------
    Fitted MultiTaskGP model in eval mode.

    Reference
    ---------
    Bonilla, E.V., Chai, K.M.A., & Williams, C.K.I. (2007).
    "Multi-task Gaussian Process Prediction." NIPS 2007.
    """
    import contextlib

    import gpytorch
    from botorch.models import MultiTaskGP

    if device is None:
        device = get_device()

    # Append task index as last column (BoTorch MultiTaskGP convention)
    X_aug = np.column_stack([X_train, task_indices.reshape(-1, 1)])
    X = _to_tensor(X_aug, device)
    y = _to_tensor(y_train.ravel(), device).unsqueeze(-1)

    n_tasks = len(np.unique(task_indices))
    effective_rank = min(rank, n_tasks)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
    )
    likelihood.noise = 0.5

    model = MultiTaskGP(
        train_X=X,
        train_Y=y,
        task_feature=-1,
        rank=effective_rank,
        likelihood=likelihood,
        all_tasks=sorted(set(task_indices.tolist())),
    )
    model.to(device)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with contextlib.suppress(RuntimeError, ValueError):
            fit_gpytorch_mll(mll)

    model.eval()
    return model


def fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_sparse: bool = False,
    n_inducing: int = 512,
    device: torch.device | None = None,
    _rng: np.random.RandomState | None = None,
    kernel_type: str = "matern",
    kernel_kwargs: dict | None = None,
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
    kernel_type : Kernel function. "matern" (default), "tanimoto",
        "aitchison", "dkl", "rf", "compositional", or "robust".
    kernel_kwargs : dict, optional
        Extra arguments passed to kernel constructors. For
        ``kernel_type="compositional"``, must include ``fp_indices``,
        ``ratio_indices``, and ``synth_indices`` (lists of int).

    Returns
    -------
    Fitted GP model in eval mode.
    """
    if kernel_type not in KERNEL_TYPES:
        raise ValueError(f"Unknown kernel_type {kernel_type!r}. Valid options: {sorted(KERNEL_TYPES)}")

    if device is None:
        device = get_device()

    X = _to_tensor(X_train, device)
    y = _to_tensor(y_train.ravel(), device).unsqueeze(-1)

    if kernel_type == "dkl":
        return _fit_dkl_gp(X, y, device)

    if kernel_type == "rf":
        return _fit_rf_kernel_gp(X, y, device)

    if kernel_type == "robust":
        return _fit_robust_gp(X, y, device)

    covar_module = None
    if kernel_type == "tanimoto":
        from gpytorch.kernels import ScaleKernel

        from .kernels import TanimotoKernel

        covar_module = ScaleKernel(TanimotoKernel())
    elif kernel_type == "aitchison":
        from gpytorch.kernels import ScaleKernel

        from .kernels import AitchisonKernel

        covar_module = ScaleKernel(AitchisonKernel())
    elif kernel_type == "compositional":
        from .kernels import CompositionalProductKernel

        kw = kernel_kwargs or {}
        fp_idx = kw.get("fp_indices", [])
        ratio_idx = kw.get("ratio_indices", [])
        synth_idx = kw.get("synth_indices", [])
        if not fp_idx and not ratio_idx and not synth_idx:
            raise ValueError(
                "kernel_type='compositional' requires at least one non-empty "
                "index list in kernel_kwargs (fp_indices, ratio_indices, synth_indices)."
            )
        covar_module = CompositionalProductKernel(
            fp_indices=fp_idx,
            ratio_indices=ratio_idx,
            synth_indices=synth_idx,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if use_sparse:
            n_ind = min(n_inducing, len(X_train))
            _rng_local = _rng if _rng is not None else np.random.RandomState()
            idx = _rng_local.choice(len(X_train), n_ind, replace=False)
            inducing = X[idx].clone()
            model = SingleTaskVariationalGP(
                X,
                y,
                inducing_points=inducing,
                covar_module=covar_module,
            )
            model.to(device)

            # Manual training loop for variational GP.
            # fit_gpytorch_mll uses L-BFGS-B which is inappropriate for
            # variational parameters — use Adam with VariationalELBO instead.
            mll = VariationalELBO(
                model.likelihood,
                model.model,
                num_data=len(X_train),
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
            model = SingleTaskGP(X, y, covar_module=covar_module)
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

    For Matern/RBF: L = sqrt(outputscale) / min(lengthscale), a conservative
    upper bound on ||d mu / dx||.

    For kernels without lengthscale (e.g. Tanimoto): returns 1.0 as a
    reasonable default. LP penalization still works but the exclusion
    radius is not calibrated to the kernel geometry. For Tanimoto
    kernels, prefer TS-Batch over LP for more reliable batch diversity.
    """
    kernel = getattr(model, "covar_module", None)
    if kernel is None:
        # SingleTaskVariationalGP wraps the kernel
        inner = getattr(model, "model", None)
        kernel = getattr(inner, "covar_module", None) if inner is not None else None
    if kernel is None:
        return 1.0

    base = getattr(kernel, "base_kernel", None)
    if base is not None and getattr(base, "lengthscale", None) is not None:
        outputscale = kernel.outputscale.detach().cpu().item()  # type: ignore[union-attr]
        lengthscale = base.lengthscale.detach().cpu().numpy().ravel()  # type: ignore[union-attr]
        return float(np.sqrt(outputscale) / np.min(lengthscale))

    ls = getattr(kernel, "lengthscale", None)
    if ls is not None:
        lengthscale = ls.detach().cpu().numpy().ravel()
        return float(1.0 / np.min(lengthscale))

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
    kernel_type: str = "matern",
    kernel_kwargs: dict | None = None,
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
    kappa : UCB exploration parameter.
    xi : EI/LogEI improvement jitter.
    seed : Random seed.
    use_sparse : Use variational GP for large training sets.
    n_inducing : Number of inducing points for sparse GP.
    kernel_type : Kernel function ("matern", "tanimoto", "compositional", etc.).
    kernel_kwargs : dict, optional
        Extra arguments for kernel construction (e.g., index lists for
        ``kernel_type="compositional"``).

    Returns
    -------
    List of selected pool_indices (length batch_size or fewer if pool exhausted).
    """
    pool_indices = np.asarray(pool_indices)
    batch_size = min(batch_size, len(X_pool))

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model = fit_gp(
        X_train,
        y_train,
        use_sparse=use_sparse,
        n_inducing=n_inducing,
        _rng=rng,
        kernel_type=kernel_type,
        kernel_kwargs=kernel_kwargs,
    )
    y_best = float(y_train.max())

    strategy = batch_strategy.lower()
    device = next(model.parameters()).device
    if strategy == "qlogei":
        return _batch_qlogei(model, X_train, X_pool, pool_indices, batch_size, device=device)
    elif strategy == "gibbon":
        return _batch_gibbon(model, X_pool, pool_indices, batch_size, device=device)
    elif strategy == "ts":
        return _batch_ts(model, X_pool, pool_indices, batch_size, seed)
    elif strategy == "kb":
        return _batch_kb(
            model,
            X_pool,
            pool_indices,
            batch_size,
            acq_type,
            y_best,
            kappa,
            xi,
            randomize=False,
        )
    elif strategy == "rkb":
        return _batch_kb(
            model,
            X_pool,
            pool_indices,
            batch_size,
            acq_type,
            y_best,
            kappa,
            xi,
            randomize=True,
        )
    elif strategy == "lp":
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
    else:
        raise ValueError(f"Unknown batch_strategy '{batch_strategy}'. Choose from: kb, rkb, lp, ts, qlogei, gibbon")


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

    for i in range(batch_size):
        scores = score_acquisition(
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

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                current_model = current_model.condition_on_observations(
                    x_new,
                    y_fantasy,
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
    pool_indices = np.asarray(pool_indices)
    batch_size = min(batch_size, len(X_pool))

    # No ratio features or non-KB batch strategies → fall back to standard discrete scoring
    # Mixed optimization is only beneficial for sequential KB/RKB with ratio dims.
    # TS, qLogEI, GIBBON, LP all work directly on the discrete pool.
    if len(ratio_indices) == 0 or batch_strategy.lower() not in ("kb", "rkb"):
        return select_batch(
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

    model = fit_gp(
        X_train,
        y_train,
        use_sparse=use_sparse,
        n_inducing=n_inducing,
        _rng=rng,
        kernel_type=kernel_type,
        kernel_kwargs=effective_kw,
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
        scores = score_acquisition(
            cur_model, X_pool[available_mask], acq_type,
            y_best_running, kappa, xi,
        )
        local_best = int(np.argmax(scores))
        return np.where(available_mask)[0][local_best]

    def _kb_condition(cur_model, global_idx, last_slot):
        """Condition model on the selected point (KB hallucination)."""
        if last_slot or isinstance(cur_model, SingleTaskVariationalGP):
            return cur_model
        x_new = _to_tensor(X_pool[global_idx].reshape(1, -1), device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_fantasy = cur_model.posterior(x_new).mean
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return cur_model.condition_on_observations(x_new, y_fantasy)

    for i in range(batch_size):
        acqf = _build_acqf(current_model, y_best_running)
        use_fallback = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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
        current_model = _kb_condition(current_model, global_idx, i == batch_size - 1)
        if i < batch_size - 1:
            x_sel = _to_tensor(X_pool[global_idx].reshape(1, -1), device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_fantasy = current_model.posterior(x_sel).mean
            y_best_running = max(y_best_running, y_fantasy.item())

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
            acqf,
            q=batch_size,
            choices=X_pool_t,
            unique=True,
        )

    return _match_candidates_to_pool(candidates, X_pool_t, X_pool, pool_indices)
