"""GP-fit backends for the GP-BO stack (DKL / RF / robust / multitask / exact / sparse).

Split out of ``gp_bo.py``. torch/botorch/gpytorch are optional dependencies
loaded lazily by the parent package; importing this module implies they are
available.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

from ._gp_device import _to_tensor, get_device

KERNEL_TYPES = {"matern", "tanimoto", "aitchison", "dkl", "rf", "compositional", "robust"}


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
        """3-layer MLP: input_dim -> 64 -> 32 -> output_dim."""
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        """Wrap a feature extractor as an InputTransform, freezing its params."""
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(X)

    def preprocess_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DKL transform is not invertible")

    def equals(self, other) -> bool:
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

    Step 1: Build a temporary SingleTaskGP with the feature extractor as
    input_transform, then optimize the joint marginal log-likelihood over
    all parameters (NN weights + GP hyperparameters) using Adam.

    Step 2: Freeze the feature extractor, transform training inputs once,
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

    # Step 1: joint training with a temporary model.
    # Standardize the targets once, matching the Standardize outcome transform
    # the final SingleTaskGP applies in Step 2. Training the temp model on raw y
    # while the GP standardizes internally mismatches the MLL scale and corrupts
    # the jointly trained feature extractor.
    y_2d = y if y.dim() == 2 else y.unsqueeze(-1)
    outcome_transform = Standardize(m=y_2d.shape[-1]).to(device)
    outcome_transform.train()
    y_std_2d, _ = outcome_transform(y_2d)
    y_std_2d = y_std_2d.detach()
    y_std = y_std_2d.squeeze(-1)

    # Build GP on transformed features for initialization
    with torch.no_grad():
        X_init = feature_extractor(X)
    temp_model = SingleTaskGP(X_init, y_std_2d, outcome_transform=None)
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
        temp_model.set_train_data(X_feat, y_std, strict=False)
        try:
            output = temp_model(X_feat)
            loss = -mll(output, y_std)
        except (ValueError, RuntimeError):
            # The deep-kernel NN can diverge and drive a GP hyperparameter
            # outside its LogNormalPrior support (or produce NaN features).
            # Stop training and keep the last valid hyperparameters rather than
            # crashing the whole BO run.
            break
        if not torch.isfinite(loss):
            break
        loss.backward()
        # Clip gradients so the feature extractor cannot blow up into NaNs -
        # the common cause of the prior-support violation above.
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        optimizer.step()
        curr_loss = loss.item()
        if prev_loss - curr_loss < 1e-6:
            no_improve += 1
            if no_improve >= patience:
                break
        else:
            no_improve = 0
        prev_loss = curr_loss

    # Step 2: freeze NN and build final model with input_transform
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

    mll_final = ExactMarginalLogLikelihood(model.likelihood, model)
    # Short L-BFGS polish on GP hyperparameters only (NN is frozen).
    # Cap iterations so the polish stays "short": an unrestricted L-BFGS-B can
    # move the GP hyperparameters far from the jointly trained values.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            fit_gpytorch_mll(mll_final, optimizer_kwargs={"options": {"maxiter": 50}})
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                f"DKL GP hyperparameter polish failed ({type(e).__name__}); "
                f"falling back to jointly trained prior hyperparameters",
                stacklevel=2,
            )

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

    rf_seed is threaded from the BO run seed (select_batch -> fit_gp ->
    _fit_gp_on_device) so each BO seed grows an independent RF kernel instead
    of sharing a single fixed-42 forest across all seeds.

    The RF kernel K(x_i, x_j) = fraction of trees where x_i and x_j
    land in the same leaf. This is data-adaptive and PSD (Scornet 2016).
    The RF is fit once on (X, y); the resulting kernel matrix is used
    as the GP's covariance function.

    Because the RF kernel has no differentiable hyperparameters, GP
    hyperparameter optimization is limited to the noise variance and
    output scale (via the wrapping ScaleKernel). L-BFGS fitting may
    converge quickly or fail silently - both are acceptable since the
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

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                f"RF-kernel GP MLL fit failed ({type(e).__name__}); using "
                f"default/prior hyperparameters (noise + outputscale only)",
                stacklevel=2,
            )

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
        try:
            fit_gpytorch_mll(
                mll,
                fractions_of_outliers=[0.0, 0.05, 0.1, 0.2],
                timeout_sec=300.0,
            )
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                f"Robust GP (Relevance Pursuit) fit failed ({type(e).__name__}); "
                f"using default/prior hyperparameters for this round",
                stacklevel=2,
            )

    model.eval()
    return model


def fit_multitask_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_indices: np.ndarray,
    device: torch.device | None = None,
    rank: int = 3,
    train_Yvar: np.ndarray | None = None,
) -> Model:
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

    if device is None:
        device = get_device()

    if device.type == "cuda":
        try:
            return _fit_multitask_gp_on_device(
                X_train, y_train, task_indices, device, rank=rank,
                train_Yvar=train_Yvar,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"[GP] CUDA OOM in fit_multitask_gp; retrying on CPU ({type(e).__name__})", flush=True)
            device = torch.device("cpu")

    return _fit_multitask_gp_on_device(
        X_train, y_train, task_indices, device, rank=rank,
        train_Yvar=train_Yvar,
    )


def _fit_multitask_gp_on_device(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_indices: np.ndarray,
    device: torch.device,
    *,
    rank: int,
    train_Yvar: np.ndarray | None = None,
):
    """Body of fit_multitask_gp once device has been chosen."""
    import gpytorch
    from botorch.models import MultiTaskGP

    # Append task index as last column (BoTorch MultiTaskGP convention)
    X_aug = np.column_stack([X_train, task_indices.reshape(-1, 1)])
    X = _to_tensor(X_aug, device)
    y = _to_tensor(y_train.ravel(), device).unsqueeze(-1)

    n_tasks = len(np.unique(task_indices))
    effective_rank = min(rank, n_tasks)

    if train_Yvar is not None:
        # Per-point variance path: MultiTaskGP accepts train_Yvar in
        # modern botorch (>=0.10). When provided, the per-task noise
        # hyperparameter is replaced by a fixed per-point variance
        # supplied by the caller (e.g. ensemble std + conformal half-width).
        clipped = np.where(np.asarray(train_Yvar) > 0, train_Yvar, 1e-4)
        Yvar = _to_tensor(clipped.ravel(), device).unsqueeze(-1)
        model = MultiTaskGP(
            train_X=X,
            train_Y=y,
            train_Yvar=Yvar,
            task_feature=-1,
            rank=effective_rank,
            all_tasks=sorted(set(task_indices.tolist())),
        )
    else:
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
        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                f"Multi-task GP MLL fit failed ({type(e).__name__}); using "
                f"default/prior hyperparameters for this round",
                stacklevel=2,
            )

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
    train_Yvar: np.ndarray | None = None,
    seed: int | None = None,
) -> SingleTaskGP | SingleTaskVariationalGP:
    """Fit a GP surrogate model.

    ``seed`` (when provided) is threaded into the RF-proximity kernel so each
    BO run seed grows an independent random forest; ignored by other kernels.

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
    train_Yvar : (N,) per-point observation variance, optional.
        When provided, the GP fits with FixedNoiseGP rather than the
        default GaussianLikelihood. Each point's noise is treated as
        a known constant rather than a learned hyperparameter; this
        is the channel for downstream callers to propagate calibrated
        uncertainty (e.g. ensemble std + conformal half-width) into
        the surrogate posterior. Variances must be
        positive; values <= 0 are clipped to 1e-4. Ignored when
        ``use_sparse=True`` (variational GP has no FixedNoise variant).

    Returns
    -------
    Fitted GP model in eval mode.
    """
    if kernel_type not in KERNEL_TYPES:
        raise ValueError(f"Unknown kernel_type {kernel_type!r}. Valid options: {sorted(KERNEL_TYPES)}")

    if device is None:
        device = get_device()

    if device.type == "cuda":
        try:
            return _fit_gp_on_device(
                X_train, y_train, device,
                use_sparse=use_sparse, n_inducing=n_inducing,
                _rng=_rng, kernel_type=kernel_type, kernel_kwargs=kernel_kwargs,
                train_Yvar=train_Yvar, seed=seed,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"[GP] CUDA OOM in fit_gp; retrying on CPU ({type(e).__name__})", flush=True)
            device = torch.device("cpu")

    return _fit_gp_on_device(
        X_train, y_train, device,
        use_sparse=use_sparse, n_inducing=n_inducing,
        _rng=_rng, kernel_type=kernel_type, kernel_kwargs=kernel_kwargs,
        train_Yvar=train_Yvar, seed=seed,
    )


def _fit_gp_on_device(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    *,
    use_sparse: bool,
    n_inducing: int,
    _rng: np.random.RandomState | None,
    kernel_type: str,
    kernel_kwargs: dict | None,
    train_Yvar: np.ndarray | None = None,
    seed: int | None = None,
) -> SingleTaskGP | SingleTaskVariationalGP:
    """Body of fit_gp once device has been chosen. See fit_gp() docstring."""
    X = _to_tensor(X_train, device)
    y = _to_tensor(y_train.ravel(), device).unsqueeze(-1)
    Yvar = None
    if train_Yvar is not None:
        # Clip non-positive variances so the GaussianLikelihood does not
        # collapse to a delta function; matches the floor convention used
        # when composing per-point variances.
        clipped = np.where(np.asarray(train_Yvar) > 0, train_Yvar, 1e-4)
        Yvar = _to_tensor(clipped.ravel(), device).unsqueeze(-1)

    if kernel_type == "dkl":
        return _fit_dkl_gp(X, y, device)

    if kernel_type == "rf":
        rf_seed = 42 if seed is None else int(seed)
        return _fit_rf_kernel_gp(X, y, device, rf_seed=rf_seed)

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
            # variational parameters - use Adam with VariationalELBO instead.
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
            # SingleTaskGP accepts train_Yvar directly in modern botorch
            # (>=0.10). When train_Yvar is None it falls through to a
            # learned GaussianLikelihood; when present, BoTorch wires a
            # FixedNoiseGaussianLikelihood internally and treats each
            # point's variance as a known constant.
            model = SingleTaskGP(X, y, train_Yvar=Yvar, covar_module=covar_module)
            model.to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except (ModelFittingError, RuntimeError, ValueError) as e:
                # Some kernels (e.g. Tanimoto on near-degenerate fingerprint
                # blocks) can fail every MLL retry. Aborting would truncate the
                # whole BO run to the seed round (results below random); instead
                # fall back to the model's default/prior hyperparameters so the
                # round still produces a usable posterior and the run continues.
                warnings.warn(
                    f"GP MLL fit failed ({type(e).__name__}); using default "
                    f"hyperparameters for this round",
                    stacklevel=2,
                )

    model.eval()
    return model
