"""Bayesian optimizer for LNP formulation design.

Supports multiple surrogate models, acquisition functions, and batch
strategies through a unified API. See ``docs/surrogates.md`` for a
comprehensive reference with citations and usage guidance.

Surrogate types
---------------

**GP-based** (support all acquisition types and batch strategies):

    ``"gp"``            Gaussian Process via BoTorch (default). Kernel
                        selected by ``kernel_type``: ``"matern"`` (default),
                        ``"tanimoto"``, ``"aitchison"``, ``"dkl"``, ``"rf"``,
                        ``"compositional"``, ``"robust"``.
    ``"gp_sklearn"``    Alias for ``gp`` with ``gp_engine="sklearn"``
                        (continuous acquisition optimization).
    ``"gp_mixed"``      Mixed discrete-continuous GP (enumerate IL
                        configs, optimize ratios continuously).
    ``"robust_gp"``     Robust GP via Relevance Pursuit — automatically
                        detects and downweights outliers (Ament et al. 2024).
    ``"multitask_gp"``  Multi-task GP with ICM coregionalization across
                        studies (Bonilla et al. 2007). Requires ``study_id``
                        column in the dataset.
    ``"casmopolitan"``  CASMOPolitan mixed-variable GP with trust regions.

**Tree/ensemble** (discrete pool scoring, built-in acquisition):

    ``"xgb"``           XGBoost greedy (predicted mean, no exploration).
    ``"xgb_ucb"``       XGBoost + MAPIE conformal UCB (Barber et al. 2021).
    ``"rf_ucb"``        Random Forest tree-variance UCB.
    ``"rf_ts"``         Random Forest Thompson Sampling (per-tree draws).
    ``"ngboost"``       NGBoost distributional UCB (Duan et al. 2020).
    ``"xgb_cqr"``       XGBoost + Conformalized Quantile Regression UCB
                        (Romano et al. 2019).
    ``"ridge"``         BayesianRidge mean + kappa * std (MacKay 1992).

**Neural UQ** (discrete pool scoring with learned uncertainty):

    ``"deep_ensemble"`` Deep Ensemble (5-network) UCB
                        (Lakshminarayanan et al. 2017).
    ``"sngp"``          Spectral-Normalized Neural GP — distance-aware
                        MLP with RFF output layer (Liu et al. 2023).
    ``"laplace"``       MLP + post-hoc Laplace approximation
                        (Daxberger et al. 2021).
    ``"tabpfn"``        TabPFN zero-shot foundation model
                        (Hollmann et al. 2025).
    ``"gp_ucb"``        Sklearn GP UCB (discrete pool scoring).

**Preference / domain-robust** (discrete pool scoring):

    ``"bradley_terry"`` Pairwise preference model — learns utility from
                        within-study comparisons (Bradley & Terry 1952).
    ``"groupdro"``      GroupDRO across studies — robust to worst-group
                        shift (Sagawa et al. 2020). Uses ``study_id``.
    ``"vrex"``          V-REx risk extrapolation — penalizes cross-study
                        loss variance (Krueger et al. 2021). Uses ``study_id``.

Acquisition types (for GP-based surrogates)
-------------------------------------------
    ``"UCB"``   Upper Confidence Bound (mu + kappa * sigma)
    ``"EI"``    Expected Improvement
    ``"LogEI"`` Log Expected Improvement (numerically stable)

Batch strategies (for GP-based surrogates)
------------------------------------------
    ``"kb"``     Kriging Believer (hallucinate with posterior mean)
    ``"rkb"``    Randomized KB (hallucinate with posterior samples)
    ``"lp"``     Local Penalization (soft exclusion zones)
    ``"ts"``     Thompson Sampling (independent posterior draws)
    ``"qlogei"`` q-Log Noisy Expected Improvement (BoTorch native joint)
    ``"gibbon"`` GIBBON: information-theoretic + DPP diversity (Moss et al. 2021)

Example::

    from LNPBO.data.dataset import Dataset
    from LNPBO.space.formulation import FormulationSpace
    from LNPBO.optimization.optimizer import Optimizer

    dataset = Dataset.from_lnpdb_csv("my_screen.csv")
    encoded = dataset.encode_dataset(feature_type="lantern")
    space = FormulationSpace.from_dataset(encoded)

    # BoTorch GP (default):
    optimizer = Optimizer(space=space, candidate_pool=encoded.df)

    # Robust GP (outlier-robust):
    optimizer = Optimizer(
        space=space,
        surrogate_type="robust_gp",
        candidate_pool=encoded.df,
    )

    # SNGP with distance-aware uncertainty:
    optimizer = Optimizer(
        space=space,
        surrogate_type="sngp",
        candidate_pool=encoded.df,
    )

    # Multi-task GP across studies (requires study_id in data):
    optimizer = Optimizer(
        space=space,
        surrogate_type="multitask_gp",
        candidate_pool=encoded.df,
    )

    suggestions = optimizer.suggest(output_csv="round1.csv")
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..data.dataset import _ENC_PREFIXES
from ..space.formulation import FormulationSpace
from ._logging import logger
from .bayesopt import perform_bayesian_optimization
from .doe import mixture_doe

ENC_PREFIXES = [p + "pc" for p in _ENC_PREFIXES]
CTX_PREFIX = "ctx_"

# ---------------------------------------------------------------------------
# Surrogate registry — capabilities are defined per family, not per surrogate
# ---------------------------------------------------------------------------

_FAMILY_CAPS = {
    "gp_botorch": {"needs_pool": True, "supports_acq": True, "supports_batch": True},
    "gp_mixed": {"needs_pool": True, "supports_acq": True, "supports_batch": True},
    "gp_multitask": {"needs_pool": True, "supports_acq": True, "supports_batch": True},
    "gp_sklearn": {"needs_pool": False, "supports_acq": True, "supports_batch": False},
    "discrete": {"needs_pool": True, "supports_acq": False, "supports_batch": True},
    "casmopolitan": {"needs_pool": True, "supports_acq": True, "supports_batch": False},
}

SURROGATE_TYPES = {
    # --- GP-based surrogates ---
    "gp": "gp_botorch",
    "gp_mixed": "gp_mixed",
    "gp_sklearn": "gp_sklearn",
    "robust_gp": "gp_botorch",  # BoTorch RobustRelevancePursuitSingleTaskGP
    "multitask_gp": "gp_multitask",  # BoTorch MultiTaskGP with ICM
    "casmopolitan": "casmopolitan",
    # --- Discrete pool-scoring surrogates ---
    "xgb": "discrete",
    "xgb_ucb": "discrete",
    "rf_ucb": "discrete",
    "rf_ts": "discrete",
    "ngboost": "discrete",
    "xgb_cqr": "discrete",
    "deep_ensemble": "discrete",
    "tabpfn": "discrete",
    "ridge": "discrete",
    "gp_ucb": "discrete",
    # --- Neural UQ surrogates ---
    "sngp": "discrete",  # Spectral-Normalized Neural GP (Liu et al. 2023)
    "laplace": "discrete",  # MLP + Laplace approximation (Daxberger et al. 2021)
    # --- Preference / domain-robust surrogates ---
    "bradley_terry": "discrete",  # Pairwise preference model (Bradley & Terry 1952)
    "groupdro": "discrete",  # GroupDRO across studies (Sagawa et al. 2020)
    "vrex": "discrete",  # V-REx risk extrapolation (Krueger et al. 2021)
}

ACQUISITION_TYPES = {"UCB", "EI", "LogEI"}

BATCH_STRATEGIES = {"kb", "rkb", "lp", "ts", "qlogei", "gibbon"}
DISCRETE_BATCH_STRATEGIES = {"greedy", "ts"}
ALL_BATCH_STRATEGIES = BATCH_STRATEGIES | DISCRETE_BATCH_STRATEGIES

# Compound acquisition type mapping for the legacy sklearn GP pipeline
_SKLEARN_ACQ_MAP = {
    ("UCB", "kb"): "UCB",
    ("EI", "kb"): "EI",
    ("LogEI", "kb"): "LogEI",
    ("UCB", "lp"): "LP_UCB",
    ("EI", "lp"): "LP_EI",
    ("LogEI", "lp"): "LP_LogEI",
}


class Optimizer:
    """Bayesian optimizer for LNP formulation design.

    Iteratively suggests batches of new formulations to test, using a
    surrogate model to predict which candidates are most promising.

    Parameters
    ----------
    space : FormulationSpace
        Defines the lipid components, molar ratio bounds, and current
        dataset of evaluated formulations.

    surrogate_type : str
        The surrogate model to use for predicting formulation performance.
        Default ``"gp"`` uses a Gaussian Process (backend selected by
        ``gp_engine``). See module docstring for all options.

    gp_engine : str
        GP backend when ``surrogate_type="gp"``: ``"botorch"`` (default,
        BoTorch/GPyTorch exact GP with discrete pool scoring) or
        ``"sklearn"`` (sklearn GP with continuous acquisition optimization).
        Ignored for non-GP surrogates. ``surrogate_type="gp_sklearn"``
        is equivalent to ``surrogate_type="gp", gp_engine="sklearn"``.
        Note: ``gp_engine="sklearn"`` is not supported by
        ``suggest_indices()`` (it requires a ``FormulationSpace``).

    acquisition_type : str
        Acquisition function for GP-based surrogates: ``"UCB"`` (default),
        ``"EI"``, or ``"LogEI"``. Ignored for tree-based surrogates which
        have built-in acquisition (e.g., ``"xgb_ucb"`` always uses UCB).

    batch_strategy : str
        How to select a batch of candidates (for ``surrogate_type="gp"``):
        ``"kb"`` (default), ``"rkb"``, ``"lp"``, ``"ts"``, ``"qlogei"``.
        For discrete surrogates, use ``"greedy"`` (top-K) or ``"ts"``
        (Thompson sampling).

    kappa : float
        Exploration weight for UCB-based acquisition (default 5.0).
        Higher values favor exploration over exploitation.

    xi : float
        Exploration parameter for EI/LogEI acquisition (default 0.01).

    batch_size : int
        Number of formulations to suggest per round (default 24).

    random_seed : int
        Random seed for reproducibility (default 1).

    candidate_pool : DataFrame, optional
        Pre-encoded candidate formulations to score. Required for all
        surrogate types except ``"gp_sklearn"``.

    normalize : str
        Target normalization: ``"copula"`` (default, rank-based),
        ``"zscore"``, or ``"none"``.

    context_features : bool
        If True, include one-hot experimental context features
        (cell type, target, route of administration).
    """

    def __init__(
        self,
        space: FormulationSpace | None = None,
        surrogate_type: str = "gp",
        gp_engine: str = "botorch",
        acquisition_type: str = "UCB",
        batch_strategy: str = "kb",
        kappa: float = 5.0,
        xi: float = 0.01,
        batch_size: int = 24,
        random_seed: int = 1,
        candidate_pool: pd.DataFrame | None = None,
        normalize: str = "copula",
        context_features: bool = False,
        # gp_sklearn-specific
        alpha: float = 1e-6,
        # kernel / surrogate configuration
        kernel_type: str = "matern",
        kernel_kwargs: dict | None = None,
        surrogate_kwargs: dict | None = None,
    ):
        """Initialize Optimizer with the given configuration."""
        self.space = space
        self.surrogate_type = surrogate_type
        self.gp_engine = gp_engine
        self.acquisition_type = acquisition_type
        self.batch_strategy = batch_strategy
        self.kappa = kappa
        self.xi = xi
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.candidate_pool = candidate_pool
        self.normalize = normalize
        self.context_features = context_features
        self.alpha = alpha
        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs
        self.surrogate_kwargs = surrogate_kwargs
        self._casmopolitan_trust_region = None
        self._casmopolitan_round_start_best = None
        self._casmopolitan_restart_X = None
        self._casmopolitan_restart_y = None

        # Resolve surrogate_type="gp_sklearn" as gp + sklearn engine
        if self.surrogate_type == "gp_sklearn":
            self.surrogate_type = "gp"
            self.gp_engine = "sklearn"

        self._validate_config()

    @property
    def _family(self):
        """Effective surrogate family, accounting for gp_engine."""
        if self.surrogate_type == "gp" and self.gp_engine == "sklearn":
            return "gp_sklearn"
        return SURROGATE_TYPES[self.surrogate_type]

    def _reset_runtime_state(self) -> None:
        """Clear cross-round state that should not leak across fresh runs."""
        self._casmopolitan_trust_region = None
        self._casmopolitan_round_start_best = None
        self._casmopolitan_restart_X = None
        self._casmopolitan_restart_y = None

    def _validate_config(self):
        """Validate surrogate, acquisition, and batch strategy compatibility."""
        if self.surrogate_type not in SURROGATE_TYPES:
            raise ValueError(
                f"Unknown surrogate_type={self.surrogate_type!r}. Valid options: {sorted(SURROGATE_TYPES)}"
            )

        if self.gp_engine not in ("botorch", "sklearn"):
            raise ValueError(f"Unknown gp_engine={self.gp_engine!r}. Valid options: 'botorch', 'sklearn'")

        family = self._family
        caps = _FAMILY_CAPS[family]

        if self.acquisition_type not in ACQUISITION_TYPES:
            raise ValueError(
                f"Unknown acquisition_type={self.acquisition_type!r}. Valid options: {sorted(ACQUISITION_TYPES)}"
            )

        if not caps["supports_acq"] and self.acquisition_type != "UCB":
            raise ValueError(
                f"surrogate_type={self.surrogate_type!r} does not support "
                f"acquisition_type={self.acquisition_type!r}. "
                f"Tree-based surrogates use their built-in acquisition. "
                f"For UCB exploration, use surrogate_type='xgb_ucb' or 'rf_ucb'."
            )

        if family in ("gp_botorch", "gp_mixed", "gp_multitask"):
            if self.batch_strategy not in BATCH_STRATEGIES:
                raise ValueError(
                    f"Unknown batch_strategy={self.batch_strategy!r} for GP surrogate. "
                    f"Valid options: {sorted(BATCH_STRATEGIES)}"
                )
        elif family == "gp_sklearn":
            pass  # sklearn GP handles batch strategy via _SKLEARN_ACQ_MAP
        elif family == "discrete":
            if self.batch_strategy not in DISCRETE_BATCH_STRATEGIES and self.batch_strategy != "kb":
                # "kb" is the default; for discrete surrogates, treat it as "greedy"
                raise ValueError(
                    f"batch_strategy={self.batch_strategy!r} is only supported with "
                    f"surrogate_type='gp'. Discrete surrogates support "
                    f"batch_strategy='greedy' (top-K) or 'ts' (Thompson sampling)."
                )
        elif family == "casmopolitan" and self.batch_strategy not in ("kb", "greedy"):
            raise ValueError(
                f"surrogate_type='casmopolitan' uses internal KB batching. "
                f"batch_strategy={self.batch_strategy!r} is not supported."
            )

        if self.normalize not in ("copula", "zscore", "none"):
            raise ValueError(f"Unknown normalize={self.normalize!r}. Valid options: 'copula', 'zscore', 'none'")

        # Kernel-feature compatibility warnings
        _VALID_KERNELS = {"matern", "tanimoto", "aitchison", "dkl", "rf", "compositional", "robust"}
        if self.kernel_type not in _VALID_KERNELS and family in ("gp_botorch", "gp_mixed"):
            raise ValueError(
                f"Unknown kernel_type={self.kernel_type!r}. "
                f"Valid options: {sorted(_VALID_KERNELS)}"
            )

        # Best-effort kernel-feature compatibility warnings
        pool = getattr(self, "candidate_pool", None)
        if pool is not None and family in ("gp_botorch", "gp_mixed"):
            pool_cols = set(pool.columns)
            if self.kernel_type == "tanimoto":
                pca_cols = [c for c in pool_cols if "_pc" in c and "count_mfp" in c]
                if pca_cols:
                    warnings.warn(
                        "kernel_type='tanimoto' expects raw count fingerprints, "
                        "but PCA-reduced columns were detected (e.g. "
                        f"{pca_cols[0]!r}). Use reduction='none' when encoding.",
                        stacklevel=2,
                    )
            if self.kernel_type == "aitchison":
                molratio_cols = [c for c in pool_cols if c.endswith("_molratio")]
                if not molratio_cols:
                    warnings.warn(
                        "kernel_type='aitchison' expects _molratio columns for "
                        "compositional data, but none were found in candidate_pool.",
                        stacklevel=2,
                    )
            if self.kernel_type == "compositional":
                kw = getattr(self, "kernel_kwargs", None) or {}
                if "fp_indices" not in kw or "ratio_indices" not in kw:
                    warnings.warn(
                        "kernel_type='compositional' expects 'fp_indices' and "
                        "'ratio_indices' in kernel_kwargs. Missing keys may "
                        "cause errors during GP fitting.",
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def suggest(self, output_csv: str | None = None) -> pd.DataFrame:
        """Suggest the next batch of formulations to test.

        Returns a DataFrame of all previous formulations plus the new
        suggested batch (with ``Experiment_value=NaN``).
        """
        if self.space is None:
            raise ValueError("space is required for suggest(). Pass space= at construction or use suggest_indices().")
        family = self._family
        if family == "gp_sklearn":
            return self._suggest_gp_sklearn(output_csv)
        elif family in ("gp_botorch", "gp_mixed", "gp_multitask", "discrete", "casmopolitan"):
            return self._suggest_pool_based(output_csv)
        else:
            raise ValueError(f"Unknown family: {family!r}")

    # ------------------------------------------------------------------
    # Low-level batch selection on pre-split indices
    # ------------------------------------------------------------------

    def suggest_indices(
        self,
        df,
        feature_cols,
        training_idx,
        pool_idx,
        round_num=0,
        encoded_dataset=None,
        batch_size=None,
    ):
        """Select batch indices from a pre-split pool.

        Low-level API used by the benchmark runner. Operates on a
        DataFrame with pre-computed feature columns and explicit
        training/pool index splits, without requiring a
        ``FormulationSpace`` or ``candidate_pool``.

        Not supported for ``gp_engine="sklearn"``.

        Returns a list of selected indices from *pool_idx*.
        """
        family = self._family
        if family == "gp_sklearn":
            raise ValueError(
                "suggest_indices() does not support gp_engine='sklearn'. "
                "Use suggest() with a FormulationSpace instead."
            )

        bs = batch_size if batch_size is not None else self.batch_size
        seed = self.random_seed + round_num

        if family == "casmopolitan" and round_num == 0:
            self._reset_runtime_state()

        # Prospective PLS refit
        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=df)

        X_train, y_train, X_pool, pool_arr = self._prepare_indices_data(
            df, feature_cols, training_idx, pool_idx,
        )
        if len(X_train) == 0:
            logger.warning("No training data available — returning empty batch")
            return []
        if len(X_train) < 5:
            logger.warning("Small training set (N=%d) — surrogate may be unreliable", len(X_train))
        if len(X_pool) < bs:
            logger.warning("Pool exhausted (%d < batch_size=%d) — returning remaining candidates", len(X_pool), bs)
            return list(range(len(X_pool))) if len(X_pool) > 0 else []

        # For benchmark reproducibility, force single-threaded surrogates
        sk = dict(self.surrogate_kwargs or {})
        sk.setdefault("n_jobs", 1)

        il_names = None
        if family == "casmopolitan":
            il_names = (df.loc[training_idx, "IL_name"], df.loc[pool_idx, "IL_name"])

        # Extract group/task info for surrogates that need it
        group_ids = None
        task_indices = None
        if self.surrogate_type in ("groupdro", "vrex", "bradley_terry", "multitask_gp"):
            if "study_id" in df.columns:
                train_study = df.loc[training_idx, "study_id"].values
                if self.surrogate_type == "multitask_gp":
                    unique_studies = sorted(set(train_study))
                    study_to_int = {s: i for i, s in enumerate(unique_studies)}
                    task_indices = np.array([study_to_int[s] for s in train_study])
                else:
                    group_ids = train_study

        return self._run_batch_selection(
            X_train, y_train, X_pool, pool_arr,
            batch_size=bs, seed=seed, surrogate_kwargs=sk,
            il_names=il_names, group_ids=group_ids,
            task_indices=task_indices,
        )

    # ------------------------------------------------------------------
    # Shared data preparation and batch selection
    # ------------------------------------------------------------------

    def _prepare_indices_data(self, df, feature_cols, training_idx, pool_idx):
        """Extract and clean train/pool arrays from *df*, applying normalization and NaN/inf filtering."""
        from ._normalize import normalize_values

        X_train = df.loc[training_idx, feature_cols].values.astype(np.float64)
        y_train = normalize_values(
            df.loc[training_idx, "Experiment_value"].values.astype(np.float64),
            self.normalize,
        )
        X_pool = df.loc[pool_idx, feature_cols].values.astype(np.float64)
        pool_arr = np.array(pool_idx)

        # Drop NaN/inf training rows
        valid_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        # Drop NaN/inf pool rows
        valid_pool = np.isfinite(X_pool).all(axis=1)
        X_pool = X_pool[valid_pool]
        pool_arr = pool_arr[valid_pool]

        return X_train, y_train, X_pool, pool_arr

    def _run_batch_selection(
        self, X_train, y_train, X_pool, pool_indices, *,
        batch_size, seed, surrogate_kwargs=None, il_names=None,
        group_ids=None, task_indices=None,
    ):
        """Dispatch batch selection to the appropriate surrogate family.

        Single entry point for all batch selection logic, called by both
        ``suggest()`` and ``suggest_indices()``. Returns a list of
        selected values from *pool_indices*.
        """
        family = self._family
        sk = surrogate_kwargs if surrogate_kwargs is not None else self.surrogate_kwargs

        if family in ("gp_botorch", "gp_mixed"):
            use_sparse = len(X_train) > 1000
            # robust_gp uses a different model class but the same batch selection
            effective_kernel = self.kernel_type
            if self.surrogate_type == "robust_gp":
                effective_kernel = "robust"

            if family == "gp_mixed":
                from .gp_bo import select_batch_mixed

                kw = self.kernel_kwargs or {}
                selected = select_batch_mixed(
                    X_train, y_train, X_pool, pool_indices,
                    batch_size=batch_size,
                    acq_type=self.acquisition_type,
                    batch_strategy=self.batch_strategy,
                    fp_indices=kw.get("fp_indices", []),
                    ratio_indices=kw.get("ratio_indices", []),
                    synth_indices=kw.get("synth_indices", []),
                    kappa=self.kappa, xi=self.xi, seed=seed,
                    use_sparse=use_sparse,
                    kernel_type=effective_kernel,
                    kernel_kwargs=self.kernel_kwargs,
                )
            else:
                from .gp_bo import select_batch

                selected = select_batch(
                    X_train, y_train, X_pool, pool_indices,
                    batch_size=batch_size,
                    acq_type=self.acquisition_type,
                    batch_strategy=self.batch_strategy,
                    kappa=self.kappa, xi=self.xi, seed=seed,
                    use_sparse=use_sparse,
                    kernel_type=effective_kernel,
                    kernel_kwargs=self.kernel_kwargs,
                )
            return list(selected)

        if family == "gp_multitask":
            from .gp_bo import fit_multitask_gp, score_acquisition

            if task_indices is None:
                raise ValueError(
                    "surrogate_type='multitask_gp' requires study_id column in the "
                    "training data. Ensure your dataset includes study_id."
                )
            model = fit_multitask_gp(X_train, y_train, task_indices)
            # Score the pool (append a dummy task index — use the most common
            # training task for prediction context)
            from collections import Counter
            most_common_task = Counter(task_indices.tolist()).most_common(1)[0][0]
            X_pool_aug = np.column_stack([
                X_pool,
                np.full(len(X_pool), most_common_task),
            ])
            scores = score_acquisition(
                model, X_pool_aug, self.acquisition_type,
                float(y_train.max()), self.kappa, self.xi,
            )
            top = np.argsort(scores)[-batch_size:][::-1]
            return [pool_indices[i] for i in top]

        if family == "discrete":
            from .discrete import score_candidate_pool, score_candidate_pool_ts_batch

            # Discrete surrogates only support "ts" (Thompson sampling) or greedy
            # top-K scoring. "kb" (the default) is treated as greedy because
            # kriging believer requires GP fantasization which discrete surrogates
            # don't support.
            effective_strategy = self.batch_strategy if self.batch_strategy == "ts" else "greedy"
            scoring_fn = score_candidate_pool_ts_batch if effective_strategy == "ts" else score_candidate_pool
            top_k, _ = scoring_fn(
                X_train, y_train, X_pool,
                surrogate=self.surrogate_type,
                batch_size=batch_size,
                kappa=self.kappa,
                random_seed=seed,
                surrogate_kwargs=sk,
                group_ids=group_ids,
            )
            return [pool_indices[i] for i in top_k]

        if family == "casmopolitan":
            from .casmopolitan import _append_restart_observation, select_pool_batch_casmopolitan

            acq_func = self.acquisition_type.lower()
            if acq_func == "logei":
                acq_func = "ei"

            # Build IL categorical encoding
            il_train, il_pool = il_names
            all_il = pd.concat([il_train, il_pool], ignore_index=True)
            il_map = {name: i for i, name in enumerate(all_il.unique())}
            il_cat_train = il_train.map(il_map).values.reshape(-1, 1)
            il_cat_pool = il_pool.map(il_map).values.reshape(-1, 1)

            X_train_aug = np.column_stack([il_cat_train, X_train])
            X_pool_aug = np.column_stack([il_cat_pool, X_pool])
            cont_indices = list(range(1, X_train_aug.shape[1]))

            current_best = float(np.max(y_train))
            restart_from_archive = False
            if self._casmopolitan_trust_region is not None and self._casmopolitan_round_start_best is not None:
                improved = current_best > self._casmopolitan_round_start_best
                restart_from_archive = self._casmopolitan_trust_region.update(improved)
                if restart_from_archive:
                    incumbent_idx = int(np.argmax(y_train))
                    self._casmopolitan_restart_X, self._casmopolitan_restart_y = _append_restart_observation(
                        self._casmopolitan_restart_X,
                        self._casmopolitan_restart_y,
                        X_train_aug[incumbent_idx],
                        current_best,
                        X_train_aug,
                        y_train,
                        np.random.RandomState(seed),
                    )

            top_indices, self._casmopolitan_trust_region = select_pool_batch_casmopolitan(
                X_train_aug, y_train, X_pool_aug,
                il_cat_train=il_cat_train.ravel(),
                il_cat_pool=il_cat_pool.ravel(),
                cont_feature_indices=cont_indices,
                cat_feature_indices=[0],
                batch_size=batch_size,
                kappa=self.kappa,
                random_seed=seed,
                acq_func=acq_func,
                trust_region=self._casmopolitan_trust_region,
                restart_from_archive=restart_from_archive,
                restart_X_raw=self._casmopolitan_restart_X,
                restart_y=self._casmopolitan_restart_y,
                restart_kappa=self.kappa,
            )
            self._casmopolitan_round_start_best = current_best
            return [pool_indices[i] for i in top_indices]

        raise ValueError(f"_run_batch_selection does not support family={family!r}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_feature_cols(self, df):
        """Identify feature columns from encoding prefixes and variable ratios."""
        feature_cols = []
        for role in ["IL", "HL", "CHL", "PEG"]:
            for prefix in ENC_PREFIXES:
                role_cols = [c for c in df.columns if c.startswith(f"{role}_{prefix}")]
                feature_cols.extend(sorted(role_cols))
        for role in ["IL", "HL", "CHL", "PEG"]:
            col = f"{role}_molratio"
            if col in df.columns and df[col].nunique() > 1:
                feature_cols.append(col)
        if "IL_to_nucleicacid_massratio" in df.columns and df["IL_to_nucleicacid_massratio"].nunique() > 1:
            feature_cols.append("IL_to_nucleicacid_massratio")
        return feature_cols

    def _prepare_pool(self, dataset, feature_cols):
        """Prepare training and pool data from *dataset* + ``candidate_pool``.

        Excludes evaluated formulations, deduplicates by composition,
        drops NaN/inf rows, and optionally adds context features.
        Returns ``(train_df, pool_df, feature_cols)``.
        """
        if self.candidate_pool is None:
            raise ValueError(
                f"candidate_pool is required for surrogate_type={self.surrogate_type!r}. "
                f"Pass candidate_pool=encoded.df to use the dataset itself as the pool."
            )

        ctx_levels = None
        train_source = dataset.df
        if self.context_features:
            from ..data.context import encode_context

            train_source, ctx_cols, ctx_levels = encode_context(dataset.df.copy())
            feature_cols = [*feature_cols, *ctx_cols]

        # Exclude already-evaluated formulations from pool.
        # When candidate_pool is a superset of training data (typical),
        # this removes the training rows. When candidate_pool IS the
        # training data (e.g., notebooks passing encoded.df as pool),
        # we still need candidates, so we only exclude formulations
        # that have been evaluated AND were selected by BO (Round > 0).
        if "Formulation_ID" in train_source.columns:
            evaluated_mask = train_source["Experiment_value"].notna()
            evaluated_ids = set(train_source.loc[evaluated_mask, "Formulation_ID"].dropna().astype(int))
            pool_df = self.candidate_pool[~self.candidate_pool["Formulation_ID"].isin(evaluated_ids)].copy()

            # If pool is empty, the candidate_pool likely IS the training data.
            # Fall back: only exclude BO-selected formulations (Round > 0).
            if pool_df.empty and "Round" in train_source.columns:
                bo_mask = train_source["Round"].fillna(0).astype(int) > 0
                bo_ids = set(train_source.loc[bo_mask, "Formulation_ID"].dropna().astype(int))
                pool_df = self.candidate_pool[~self.candidate_pool["Formulation_ID"].isin(bo_ids)].copy()
        else:
            pool_df = self.candidate_pool.copy()

        if pool_df.empty:
            raise ValueError("No candidates remaining in pool after excluding evaluated formulations.")

        if self.context_features and ctx_levels is not None:
            from ..data.context import encode_context

            pool_df, _, _ = encode_context(pool_df, levels=ctx_levels)

        missing = [c for c in feature_cols if c not in pool_df.columns]
        if missing:
            raise ValueError(f"Candidate pool missing feature columns: {missing}")

        # Clean training data (NaN or inf in features)
        train_features = train_source[feature_cols]
        train_mask = train_features.notna().all(axis=1) & np.isfinite(train_features.values).all(axis=1)
        if not train_mask.all():
            n_drop = int((~train_mask).sum())
            logger.info("Dropped %d training rows with missing/inf features", n_drop)
        train_df = train_source.loc[train_mask].copy()

        # Deduplicate pool
        composition_cols = [
            c
            for c in [
                "IL_name",
                "HL_name",
                "CHL_name",
                "PEG_name",
                "IL_molratio",
                "HL_molratio",
                "CHL_molratio",
                "PEG_molratio",
                "IL_to_nucleicacid_massratio",
            ]
            if c in pool_df.columns
        ]
        if composition_cols:
            n_before = len(pool_df)
            pool_df = pool_df.drop_duplicates(subset=composition_cols).reset_index(drop=True)
            n_deduped = n_before - len(pool_df)
            if n_deduped > 0:
                logger.info("Deduplicated pool: %s -> %s (%s duplicates removed)", f"{n_before:,}", f"{len(pool_df):,}", f"{n_deduped:,}")

        # Clean pool (NaN or inf in features)
        pool_features = pool_df[feature_cols]
        pool_mask = pool_features.notna().all(axis=1) & np.isfinite(pool_features.values).all(axis=1)
        if not pool_mask.all():
            n_drop = int((~pool_mask).sum())
            logger.info("Dropped %d pool rows with missing/inf features", n_drop)
            pool_df = pool_df.loc[pool_mask].copy()

        if pool_df.empty or train_df.empty:
            raise ValueError("No valid rows available after dropping missing features.")

        return train_df, pool_df, feature_cols

    def _build_result(self, dataset, selected_df, output_csv):
        """Combine existing data with new suggestions, assign IDs/Round, optionally write CSV."""
        round_number = dataset.max_round() + 1
        df_old = dataset.df.copy()

        if "Formulation_ID" not in df_old.columns:
            df_old["Formulation_ID"] = np.nan
        start_id = int(df_old["Formulation_ID"].max()) if df_old["Formulation_ID"].notna().any() else 0

        selected_df = selected_df.copy()
        selected_df["Formulation_ID"] = np.arange(start_id + 1, start_id + 1 + len(selected_df))
        selected_df["Round"] = round_number
        selected_df["Experiment_value"] = np.nan

        all_cols = sorted(set(df_old.columns) | set(selected_df.columns))
        df_old = df_old.reindex(columns=all_cols)
        selected_df = selected_df.reindex(columns=all_cols)
        df_final = pd.concat([df_old, selected_df], ignore_index=True)
        df_final = self._order_columns(df_final)

        if output_csv is not None:
            df_final.to_csv(output_csv, index=False)
            logger.info("Suggested formulations written to %s", output_csv)

        return df_final

    # ------------------------------------------------------------------
    # BoTorch/GPyTorch GP path (new pipeline)
    # ------------------------------------------------------------------

    def _suggest_pool_based(self, output_csv):
        """Suggest a batch using any pool-based surrogate (GP, discrete, CASMOPolitan)."""
        from ._normalize import normalize_values

        dataset = self.space._dataset
        assert dataset is not None
        family = self._family
        if family == "casmopolitan" and dataset.max_round() <= 0:
            self._reset_runtime_state()

        feature_cols = self._get_feature_cols(dataset.df)
        train_df, pool_df, feature_cols = self._prepare_pool(dataset, feature_cols)

        # Prospective PLS refit (GP paths only)
        if family in ("gp_botorch", "gp_mixed") and getattr(dataset, "raw_fingerprints", None):
            training_idx = train_df.index.tolist()
            dataset.refit_pls(training_idx, external_df=train_df)

        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = normalize_values(train_df["Experiment_value"].values.astype(np.float64), self.normalize)
        X_pool = pool_df[feature_cols].values.astype(np.float64)
        pool_indices = np.arange(len(pool_df))

        # CASMOPolitan needs IL name series for categorical encoding
        il_names = (train_df["IL_name"], pool_df["IL_name"]) if family == "casmopolitan" else None

        # Extract group/task info for surrogates that need it
        group_ids = None
        task_indices = None
        if self.surrogate_type in ("groupdro", "vrex", "bradley_terry", "multitask_gp"):
            if "study_id" in train_df.columns:
                train_study = train_df["study_id"].values
                if self.surrogate_type == "multitask_gp":
                    unique_studies = sorted(set(train_study))
                    study_to_int = {s: i for i, s in enumerate(unique_studies)}
                    task_indices = np.array([study_to_int[s] for s in train_study])
                else:
                    group_ids = train_study

        selected_pool_idx = self._run_batch_selection(
            X_train, y_train, X_pool, pool_indices,
            batch_size=self.batch_size, seed=self.random_seed,
            il_names=il_names, group_ids=group_ids,
            task_indices=task_indices,
        )

        selected = pool_df.iloc[selected_pool_idx].copy()
        return self._build_result(dataset, selected, output_csv)

    # ------------------------------------------------------------------
    # Legacy sklearn GP path
    # ------------------------------------------------------------------

    def _suggest_gp_sklearn(self, output_csv):
        """Suggest a batch using the legacy sklearn GP pipeline.

        Uses continuous acquisition optimization with L-BFGS-B, then
        decodes optimized feature vectors back to named components via
        nearest-neighbor matching. Falls back to mixture DoE if no
        variable components or ratios are detected.
        """
        dataset = self.space._dataset
        assert dataset is not None

        round_number = dataset.max_round() + 1

        meta = dataset.metadata
        any_components = any(meta["variable_components"].values())
        variable_ratios = sum(meta["variable_molratios"].values())

        if not any_components and variable_ratios < 2:
            logger.warning("No BO variables detected -> falling back to DoE")
            df_batch = pd.DataFrame(
                mixture_doe(
                    n_samples=self.batch_size,
                    components=["IL", "HL", "CHL", "PEG"],
                    bounds=self.space.molratio_bounds,
                    seed=self.random_seed,
                )
            )
        else:
            compound_acq = _SKLEARN_ACQ_MAP.get(
                (self.acquisition_type, self.batch_strategy),
                self.acquisition_type,
            )

            df_batch = perform_bayesian_optimization(
                data=dataset.df,
                formulation_space=self.space,
                round_number=round_number,
                acq_type=compound_acq,
                BATCH_SIZE=self.batch_size,
                RANDOM_STATE_SEED=self.random_seed,
                KAPPA=self.kappa,
                XI=self.xi,
                ALPHA=self.alpha,
            )

        df_old = dataset.df.copy()

        def decode_component(df_new, role, space):
            """Match each row's feature vector to the nearest known component for *role*."""
            configs = space.get_configs()
            comp_params = [p for p in configs["parameters"] if p["type"] == "ComponentParameter" and p["name"] == role]
            name_col = f"{role}_name"
            smiles_col = f"{role}_SMILES"

            if not comp_params:
                df_new[name_col] = dataset.df[name_col].iloc[0]
                if smiles_col in dataset.df.columns:
                    df_new[smiles_col] = dataset.df[smiles_col].iloc[0]
                return df_new

            p = comp_params[0]
            feat_cols = p["columns"]

            ref_cols = [*feat_cols, name_col]
            if smiles_col in dataset.df.columns:
                ref_cols.append(smiles_col)

            ref = dataset.df[ref_cols].dropna().drop_duplicates(subset=feat_cols).reset_index(drop=True)

            X_ref = ref[feat_cols].to_numpy()
            names = ref[f"{role}_name"].tolist()
            smiles = ref[smiles_col].tolist() if smiles_col in ref.columns else None

            def decode_row(row):
                x = row[feat_cols].to_numpy().reshape(1, -1)
                idx = pairwise_distances(x, X_ref).argmin()
                if smiles is not None:
                    return names[idx], smiles[idx]
                return names[idx], np.nan

            decoded = df_new.apply(decode_row, axis=1, result_type="expand")
            df_new[name_col] = decoded.iloc[:, 0]
            df_new[smiles_col] = decoded.iloc[:, 1]
            return df_new

        for role in ["IL", "HL", "CHL", "PEG"]:
            df_batch = decode_component(df_batch, role, self.space)

        if not df_old.empty:
            ratio_cols = [c for c in df_batch.columns if c.endswith("_molratio")]
            ratio_cols.append("IL_to_nucleicacid_massratio")
            for col in ratio_cols:
                if col in df_old.columns:
                    reference_value = df_old[col].iloc[0]
                    if col in df_batch.columns:
                        df_batch[col] = df_batch[col].fillna(reference_value)
                    else:
                        df_batch[col] = reference_value

        return self._build_result(dataset, df_batch, output_csv)

    # ------------------------------------------------------------------
    # Column ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _order_columns(df_final: pd.DataFrame) -> pd.DataFrame:
        """Reorder DataFrame columns into canonical display order."""

        def role_block(role: str):
            return [
                f"{role}_name",
                f"{role}_SMILES",
                f"{role}_molratio",
            ]

        ordered_cols = []
        ordered_cols += ["Formulation_ID", "Round"]
        ordered_cols += [
            "IL_name",
            "IL_SMILES",
            "IL_molratio",
            "IL_to_nucleicacid_massratio",
        ]
        for role in ["HL", "CHL", "PEG"]:
            ordered_cols += role_block(role)
        ordered_cols += ["Experiment_value"]

        for role in ["IL", "HL", "CHL", "PEG"]:
            enc_cols = [c for c in df_final.columns if any(c.startswith(f"{role}_{p}") for p in _ENC_PREFIXES)]
            ordered_cols += sorted(enc_cols)

        remaining = [c for c in df_final.columns if c not in ordered_cols]
        ordered_cols += remaining

        ordered_cols = [c for c in ordered_cols if c in df_final.columns]
        return df_final[ordered_cols]
