"""Bayesian optimizer for LNP formulation design.

Supports multiple surrogate models, acquisition functions, and batch
strategies through a unified API.

Surrogate types:
    ``"gp"``            BoTorch/GPyTorch exact GP (default, best for <1000 training points)
    ``"gp_sklearn"``    Legacy sklearn GP with continuous optimization
    ``"xgb"``           XGBoost greedy (predicted mean, no exploration)
    ``"xgb_ucb"``       XGBoost + MAPIE conformal UCB
    ``"rf_ucb"``        Random Forest tree-variance UCB
    ``"rf_ts"``         Random Forest Thompson Sampling (per-tree draws)
    ``"ngboost"``       NGBoost distributional UCB
    ``"xgb_cqr"``       XGBoost + Conformalized Quantile Regression UCB
    ``"deep_ensemble"`` Deep Ensemble (5-network) UCB
    ``"tabpfn"``        TabPFN zero-shot foundation model
    ``"gp_ucb"``        Sklearn GP UCB (discrete pool scoring)
    ``"casmopolitan"``  Mixed-variable GP with trust regions (CASMOPolitan)

Acquisition types (for GP-based surrogates):
    ``"UCB"``   Upper Confidence Bound (mu + kappa * sigma)
    ``"EI"``    Expected Improvement
    ``"LogEI"`` Log Expected Improvement (numerically stable)

Batch strategies (for ``surrogate_type="gp"``):
    ``"kb"``     Kriging Believer (hallucinate with posterior mean)
    ``"rkb"``    Randomized KB (hallucinate with posterior samples)
    ``"lp"``     Local Penalization (soft exclusion zones)
    ``"ts"``     Thompson Sampling (independent posterior draws)
    ``"qlogei"`` q-Log Noisy Expected Improvement (BoTorch native joint)

Example::

    from LNPBO.data.dataset import Dataset
    from LNPBO.space.formulation import FormulationSpace
    from LNPBO.optimization.optimizer import Optimizer

    dataset = Dataset.from_lnpdb_csv("my_screen.csv")
    encoded = dataset.encode_dataset(feature_type="lantern")
    space = FormulationSpace.from_dataset(encoded)

    optimizer = Optimizer(
        space=space,
        surrogate_type="xgb_ucb",
        candidate_pool=encoded.df,
    )
    suggestions = optimizer.suggest(output_csv="round1.csv")
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..data.dataset import _ENC_PREFIXES
from ..space.formulation import FormulationSpace
from .bayesopt import perform_bayesian_optimization
from .doe import mixture_doe

ENC_PREFIXES = [p + "pc" for p in _ENC_PREFIXES]
CTX_PREFIX = "ctx_"

# ---------------------------------------------------------------------------
# Surrogate registry — capabilities are defined per family, not per surrogate
# ---------------------------------------------------------------------------

_FAMILY_CAPS = {
    "gp_botorch":  {"needs_pool": True,  "supports_acq": True,  "supports_batch": True},
    "gp_sklearn":  {"needs_pool": False, "supports_acq": True,  "supports_batch": False},
    "discrete":    {"needs_pool": True,  "supports_acq": False, "supports_batch": True},
    "casmopolitan":{"needs_pool": True,  "supports_acq": True,  "supports_batch": False},
}

SURROGATE_TYPES = {
    "gp":              "gp_botorch",
    "gp_sklearn":      "gp_sklearn",
    "xgb":             "discrete",
    "xgb_ucb":         "discrete",
    "rf_ucb":          "discrete",
    "rf_ts":           "discrete",
    "ngboost":         "discrete",
    "xgb_cqr":         "discrete",
    "deep_ensemble":   "discrete",
    "tabpfn":          "discrete",
    "gp_ucb":          "discrete",
    "casmopolitan":    "casmopolitan",
}

ACQUISITION_TYPES = {"UCB", "EI", "LogEI"}

BATCH_STRATEGIES = {"kb", "rkb", "lp", "ts", "qlogei"}
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
        Default ``"gp"`` uses a BoTorch/GPyTorch Gaussian Process.
        See module docstring for all options.

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
        space: FormulationSpace,
        surrogate_type: str = "gp",
        acquisition_type: str = "UCB",
        batch_strategy: str = "kb",
        kappa: float = 5.0,
        xi: float = 0.01,
        batch_size: int = 24,
        random_seed: int = 1,
        candidate_pool: pd.DataFrame | None = None,
        normalize: str = "copula",
        context_features: bool = False,
        # gp_sklearn-specific (legacy)
        alpha: float = 1e-6,
    ):
        self.space = space
        self.surrogate_type = surrogate_type
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

        self._validate_config()

    def _validate_config(self):
        if self.surrogate_type not in SURROGATE_TYPES:
            raise ValueError(
                f"Unknown surrogate_type={self.surrogate_type!r}. "
                f"Valid options: {sorted(SURROGATE_TYPES)}"
            )

        family = SURROGATE_TYPES[self.surrogate_type]
        caps = _FAMILY_CAPS[family]

        if self.acquisition_type not in ACQUISITION_TYPES:
            raise ValueError(
                f"Unknown acquisition_type={self.acquisition_type!r}. "
                f"Valid options: {sorted(ACQUISITION_TYPES)}"
            )

        if not caps["supports_acq"] and self.acquisition_type != "UCB":
            raise ValueError(
                f"surrogate_type={self.surrogate_type!r} does not support "
                f"acquisition_type={self.acquisition_type!r}. "
                f"Tree-based surrogates use their built-in acquisition. "
                f"For UCB exploration, use surrogate_type='xgb_ucb' or 'rf_ucb'."
            )

        if family == "gp_botorch":
            if self.batch_strategy not in BATCH_STRATEGIES:
                raise ValueError(
                    f"Unknown batch_strategy={self.batch_strategy!r} for GP surrogate. "
                    f"Valid options: {sorted(BATCH_STRATEGIES)}"
                )
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
            raise ValueError(
                f"Unknown normalize={self.normalize!r}. "
                f"Valid options: 'copula', 'zscore', 'none'"
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def suggest(self, output_csv: str | None = None) -> pd.DataFrame:
        """Suggest the next batch of formulations to test.

        Returns a DataFrame containing all previous formulations plus the
        new suggested batch (with ``Experiment_value=NaN``).
        """
        family = SURROGATE_TYPES[self.surrogate_type]
        if family == "gp_sklearn":
            return self._suggest_gp_sklearn(output_csv)
        elif family == "gp_botorch":
            return self._suggest_gp_botorch(output_csv)
        elif family == "discrete":
            return self._suggest_discrete(output_csv)
        elif family == "casmopolitan":
            return self._suggest_casmopolitan(output_csv)
        else:
            raise ValueError(f"Unknown family: {family!r}")

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
        if ("IL_to_nucleicacid_massratio" in df.columns
                and df["IL_to_nucleicacid_massratio"].nunique() > 1):
            feature_cols.append("IL_to_nucleicacid_massratio")
        return feature_cols

    def _prepare_pool(self, dataset, feature_cols):
        """Prepare training and pool data from dataset + candidate_pool."""
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

        # Exclude already-evaluated formulations from pool
        evaluated_ids = set(train_source["Formulation_ID"].dropna().astype(int))
        pool_df = self.candidate_pool[
            ~self.candidate_pool["Formulation_ID"].isin(evaluated_ids)
        ].copy()

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
            print(f"  Dropped {n_drop} training rows with missing/inf features")
        train_df = train_source.loc[train_mask].copy()

        # Deduplicate pool
        composition_cols = [c for c in [
            "IL_name", "HL_name", "CHL_name", "PEG_name",
            "IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio",
            "IL_to_nucleicacid_massratio",
        ] if c in pool_df.columns]
        if composition_cols:
            n_before = len(pool_df)
            pool_df = pool_df.drop_duplicates(subset=composition_cols).reset_index(drop=True)
            n_deduped = n_before - len(pool_df)
            if n_deduped > 0:
                print(f"  Deduplicated pool: {n_before:,} -> {len(pool_df):,} ({n_deduped:,} duplicates removed)")

        # Clean pool (NaN or inf in features)
        pool_features = pool_df[feature_cols]
        pool_mask = pool_features.notna().all(axis=1) & np.isfinite(pool_features.values).all(axis=1)
        if not pool_mask.all():
            n_drop = int((~pool_mask).sum())
            print(f"  Dropped {n_drop} pool rows with missing/inf features")
            pool_df = pool_df.loc[pool_mask].copy()

        if pool_df.empty or train_df.empty:
            raise ValueError("No valid rows available after dropping missing features.")

        return train_df, pool_df, feature_cols

    def _build_result(self, dataset, selected_df, output_csv):
        """Assemble final DataFrame: old data + new suggestions."""
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
            print(f"Suggested formulations written to {output_csv}")

        return df_final

    # ------------------------------------------------------------------
    # BoTorch/GPyTorch GP path (new pipeline)
    # ------------------------------------------------------------------

    def _suggest_gp_botorch(self, output_csv):
        from ._normalize import normalize_values
        from .gp_bo import select_batch

        dataset = self.space._dataset
        assert dataset is not None

        feature_cols = self._get_feature_cols(dataset.df)
        train_df, pool_df, feature_cols = self._prepare_pool(dataset, feature_cols)

        # Prospective PLS refit
        if getattr(dataset, "raw_fingerprints", None):
            training_idx = train_df.index.tolist()
            dataset.refit_pls(training_idx, external_df=train_df)

        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = normalize_values(
            train_df["Experiment_value"].values.astype(np.float64), self.normalize
        )
        X_pool = pool_df[feature_cols].values.astype(np.float64)
        pool_indices = np.arange(len(pool_df))

        use_sparse = len(X_train) > 1000
        selected_pool_idx = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=self.batch_size,
            acq_type=self.acquisition_type,
            batch_strategy=self.batch_strategy,
            kappa=self.kappa, xi=self.xi,
            seed=self.random_seed,
            use_sparse=use_sparse,
        )

        selected = pool_df.iloc[selected_pool_idx].copy()
        return self._build_result(dataset, selected, output_csv)

    # ------------------------------------------------------------------
    # Discrete surrogate path (XGB, RF, NGBoost, etc.)
    # ------------------------------------------------------------------

    def _suggest_discrete(self, output_csv):
        from ._normalize import normalize_values
        from .discrete import score_candidate_pool, score_candidate_pool_ts_batch

        dataset = self.space._dataset
        assert dataset is not None

        feature_cols = self._get_feature_cols(dataset.df)
        train_df, pool_df, feature_cols = self._prepare_pool(dataset, feature_cols)

        X_train = train_df[feature_cols].values
        y_train = normalize_values(
            train_df["Experiment_value"].values, self.normalize
        )

        X_pool = pool_df[feature_cols].values

        # Dispatch to TS batch or greedy scoring
        use_ts = self.batch_strategy == "ts"
        if use_ts:
            scoring_fn = score_candidate_pool_ts_batch
        else:
            scoring_fn = score_candidate_pool

        top_indices, _ = scoring_fn(
            X_train, y_train, X_pool,
            surrogate=self.surrogate_type,
            batch_size=self.batch_size,
            kappa=self.kappa,
            random_seed=self.random_seed,
        )

        selected = pool_df.iloc[top_indices].copy()
        return self._build_result(dataset, selected, output_csv)

    # ------------------------------------------------------------------
    # CASMOPolitan path (mixed categorical + continuous)
    # ------------------------------------------------------------------

    def _suggest_casmopolitan(self, output_csv):
        from ._normalize import normalize_values
        from .casmopolitan import score_pool_casmopolitan

        dataset = self.space._dataset
        assert dataset is not None

        feature_cols = self._get_feature_cols(dataset.df)
        train_df, pool_df, feature_cols = self._prepare_pool(dataset, feature_cols)

        acq_func = self.acquisition_type.lower()
        if acq_func == "logei":
            acq_func = "ei"  # CASMOPolitan doesn't have LogEI; fall back to EI

        X_train_cont = train_df[feature_cols].values
        y_train = normalize_values(
            train_df["Experiment_value"].values, self.normalize
        )
        X_pool_cont = pool_df[feature_cols].values

        # Integer-encode IL identity as a categorical column
        all_il = pd.concat([train_df["IL_name"], pool_df["IL_name"]], ignore_index=True)
        il_map = {name: i for i, name in enumerate(all_il.unique())}
        il_cat_train = train_df["IL_name"].map(il_map).values.reshape(-1, 1)
        il_cat_pool = pool_df["IL_name"].map(il_map).values.reshape(-1, 1)

        # Prepend categorical IL column to feature matrix
        X_train = np.column_stack([il_cat_train, X_train_cont])
        X_pool = np.column_stack([il_cat_pool, X_pool_cont])

        cat_indices = [0]  # first column is categorical
        cont_indices = list(range(1, X_train.shape[1]))

        top_indices, _ = score_pool_casmopolitan(
            X_train, y_train, X_pool,
            il_cat_train=il_cat_train.ravel(),
            il_cat_pool=il_cat_pool.ravel(),
            cont_feature_indices=cont_indices,
            cat_feature_indices=cat_indices,
            batch_size=self.batch_size,
            kappa=self.kappa,
            random_seed=self.random_seed,
            acq_func=acq_func,
        )

        selected = pool_df.iloc[top_indices].copy()
        return self._build_result(dataset, selected, output_csv)

    # ------------------------------------------------------------------
    # Legacy sklearn GP path
    # ------------------------------------------------------------------

    def _suggest_gp_sklearn(self, output_csv):
        dataset = self.space._dataset
        assert dataset is not None

        round_number = dataset.max_round() + 1

        meta = dataset.metadata
        any_components = any(meta["variable_components"].values())
        variable_ratios = sum(meta["variable_molratios"].values())

        if not any_components and variable_ratios < 2:
            print("No BO variables detected -> falling back to DoE")
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
            enc_cols = [
                c for c in df_final.columns
                if any(c.startswith(f"{role}_{p}") for p in _ENC_PREFIXES)
            ]
            ordered_cols += sorted(enc_cols)

        remaining = [c for c in df_final.columns if c not in ordered_cols]
        ordered_cols += remaining

        ordered_cols = [c for c in ordered_cols if c in df_final.columns]
        return df_final[ordered_cols]
