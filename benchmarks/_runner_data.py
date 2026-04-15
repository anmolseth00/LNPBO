"""Data loading and feature preparation for benchmark runs."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from LNPBO.optimization.optimizer import ENC_PREFIXES


def select_warmup_seed(df, warmup_size, selection, random_seed):
    """Select a warmup seed pool and oracle split from the full dataset."""
    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(df))

    if warmup_size >= len(df):
        raise ValueError(f"warmup_size ({warmup_size}) >= dataset size ({len(df)})")

    if selection == "random":
        rng.shuffle(all_idx)
        seed_idx = sorted(all_idx[:warmup_size])
    elif selection == "bottom_75":
        threshold = df["Experiment_value"].quantile(0.75)
        bottom_mask = df["Experiment_value"] <= threshold
        bottom_idx = all_idx[bottom_mask]
        if len(bottom_idx) < warmup_size:
            seed_idx = sorted(bottom_idx)
            remaining = warmup_size - len(seed_idx)
            top_idx = all_idx[~bottom_mask]
            rng.shuffle(top_idx)
            seed_idx = sorted(list(seed_idx) + list(top_idx[:remaining]))
        else:
            rng.shuffle(bottom_idx)
            seed_idx = sorted(bottom_idx[:warmup_size])
    else:
        raise ValueError(f"Unknown warmup selection: {selection!r}")

    seed_set = set(seed_idx)
    oracle_idx = sorted([i for i in all_idx if i not in seed_set])
    return list(seed_idx), oracle_idx


class LNPDBOracle:
    """Oracle that wraps the encoded LNPDB for nearest-neighbor lookup."""

    def __init__(self, encoded_df, feature_cols):
        self.df = encoded_df.copy()
        self.feature_cols = feature_cols
        self._nn = None

    def _build_nn(self, pool_indices):
        X = self.df.loc[pool_indices, self.feature_cols].values
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X)
        return nn, pool_indices

    def lookup(self, suggestion_features, pool_indices):
        nn, idx_list = self._build_nn(pool_indices)
        idx_arr = np.array(idx_list)
        x = np.atleast_2d(suggestion_features)
        _, nn_idx = nn.kneighbors(x)
        matched_idx = idx_arr[nn_idx.ravel()]
        return matched_idx

    def get_value(self, idx):
        return self.df.loc[idx, "Experiment_value"].values


def prepare_benchmark_data(
    n_seed=500,
    random_seed=42,
    subset=None,
    reduction="pca",
    feature_type="mfp",
    n_pcs=None,
    context_features=False,
    fp_radius=None,
    fp_bits=None,
    data_df=None,
    pca_train_indices=None,
):
    """Load LNPDB, encode molecular features, and split into seed/oracle pools."""
    from LNPBO.data.dataset import Dataset
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    is_ratios_only = feature_type == "ratios_only"
    is_raw = feature_type.startswith("raw_")
    is_concat = feature_type in ("concat", "raw_concat")
    is_lantern = feature_type in ("lantern", "raw_lantern")
    is_lantern_unimol = feature_type in ("lantern_unimol", "raw_lantern_unimol")
    is_lantern_mordred = feature_type in ("lantern_mordred", "raw_lantern_mordred")
    is_chemeleon_il_only = feature_type == "chemeleon_il_only"
    is_chemeleon_helper_only = feature_type == "chemeleon_helper_only"
    effective_reduction = "none" if (is_raw or is_ratios_only) else reduction

    print(f"Loading LNPDB (reduction={effective_reduction}, features={feature_type})...")
    if data_df is None:
        dataset = load_lnpdb_full()
        df = dataset.df
    else:
        df = data_df.copy()
        if "Formulation_ID" not in df.columns:
            df["Formulation_ID"] = range(1, len(df) + 1)
        dataset = Dataset(df, source="lnpdb", name="LNPDB_benchmark")

    if subset and subset < len(df):
        df = df.sample(n=subset, random_state=random_seed).reset_index(drop=True)
        df["Formulation_ID"] = range(1, len(df) + 1)
        dataset = Dataset(df, source="lnpdb", name="LNPDB_benchmark")

    print(f"  {len(df):,} formulations loaded")
    print(f"  Experiment_value range: [{df['Experiment_value'].min():.2f}, {df['Experiment_value'].max():.2f}]")

    def _should_encode(role, n_pcs_local):
        smiles_col = f"{role}_SMILES"
        name_col = f"{role}_name"
        if df[name_col].nunique() <= 1:
            return 0
        if smiles_col not in df.columns:
            return 0
        if df[smiles_col].dropna().nunique() <= 1:
            return 0
        return n_pcs_local

    if is_raw:
        default_pcs = 2048
    elif n_pcs is not None:
        default_pcs = n_pcs
    else:
        default_pcs = 5

    il_pcs = _should_encode("IL", default_pcs)
    hl_pcs = _should_encode("HL", default_pcs if (is_raw or n_pcs is not None) else 3)
    chl_pcs = _should_encode("CHL", default_pcs if (is_raw or n_pcs is not None) else 3)
    peg_pcs = _should_encode("PEG", default_pcs if (is_raw or n_pcs is not None) else 3)

    def _role_pcs():
        return [("IL", il_pcs), ("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]

    enc = {}
    if is_ratios_only:
        print("  Ratios-only mode: no molecular encoding")
    elif is_concat:
        for role, n in _role_pcs():
            enc[role] = {"mfp": n, "unimol": n}
        print(f"  Concat (MFP+Uni-Mol) dims per role: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")
    elif is_lantern_mordred:
        for role, n in _role_pcs():
            enc[role] = {"count_mfp": n, "rdkit": n, "mordred": n}
        print(
            f"  LANTERN+Mordred (count_mfp+rdkit+mordred) dims per role:"
            f" IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}"
        )
    elif is_lantern_unimol:
        for role, n in _role_pcs():
            enc[role] = {"count_mfp": n, "rdkit": n, "unimol": n}
        print(
            f"  LANTERN+Uni-Mol (count_mfp+rdkit+unimol) dims per role:"
            f" IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}"
        )
    elif is_lantern:
        for role, n in _role_pcs():
            enc[role] = {"count_mfp": n, "rdkit": n}
        print(f"  LANTERN (count_mfp+rdkit) dims per role: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")
    elif feature_type == "lantern_il_only":
        enc["IL"] = {"count_mfp": il_pcs, "rdkit": il_pcs}
        print(f"  LANTERN IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "lantern_il_hl":
        enc["IL"] = {"count_mfp": il_pcs, "rdkit": il_pcs}
        enc["HL"] = {"count_mfp": hl_pcs, "rdkit": hl_pcs}
        print(f"  LANTERN IL+HL: IL={il_pcs}, HL={hl_pcs} PCs (CHL/PEG get ratios only)")
    elif feature_type == "lantern_il_noratios":
        enc["IL"] = {"count_mfp": il_pcs, "rdkit": il_pcs}
        print(f"  LANTERN IL no-ratios: IL={il_pcs} PCs (no molar ratios)")
    elif feature_type == "lion_il_only":
        enc["IL"] = {"lion": il_pcs}
        print(f"  LiON IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "mordred_il_only":
        enc["IL"] = {"mordred": il_pcs}
        print(f"  Mordred IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "unimol_il_only":
        enc["IL"] = {"unimol": il_pcs}
        print(f"  Uni-Mol IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "mfp_il_only":
        enc["IL"] = {"mfp": il_pcs}
        print(f"  Morgan FP IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "count_mfp_il_only":
        enc["IL"] = {"count_mfp": il_pcs}
        print(f"  Count MFP IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif is_chemeleon_il_only:
        enc["IL"] = {"chemeleon": il_pcs}
        print(f"  CheMeleon IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "agile_il_only":
        enc["IL"] = {"agile": il_pcs}
        print(f"  AGILE IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif is_chemeleon_helper_only:
        for role, n in [("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
            enc[role] = {"chemeleon": n}
        print(f"  CheMeleon helper-only: HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs} (IL gets ratios only)")
    else:
        base_type = feature_type.replace("raw_", "")
        enc_key = {
            "mfp": "mfp",
            "mordred": "mordred",
            "unimol": "unimol",
            "count_mfp": "count_mfp",
            "rdkit": "rdkit",
            "chemeleon": "chemeleon",
            "lion": "lion",
            "agile": "agile",
        }[base_type]
        for role, n in _role_pcs():
            enc[role] = {enc_key: n}
        print(f"  Encoding dims: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")

    if is_ratios_only:
        encoded = dataset
    else:
        fp_kw = {}
        if fp_radius is not None:
            fp_kw["fp_radius"] = fp_radius
        if fp_bits is not None:
            fp_kw["fp_bits"] = fp_bits

        if pca_train_indices is not None and effective_reduction != "none":
            train_df = df.iloc[pca_train_indices].copy()
            train_dataset = Dataset(train_df, source="lnpdb", name="LNPDB_pca_fit")
            train_encoded = train_dataset.encode_dataset(
                enc,
                reduction=effective_reduction,
                **fp_kw,
            )
            encoded = dataset.encode_dataset(
                enc,
                reduction=effective_reduction,
                fitted_transformers_in=train_encoded.fitted_transformers,
                **fp_kw,
            )
        else:
            encoded = dataset.encode_dataset(
                enc,
                reduction=effective_reduction,
                **fp_kw,
            )

    feature_cols = []
    for role in ["IL", "HL", "CHL", "PEG"]:
        for prefix in ENC_PREFIXES:
            role_cols = [c for c in encoded.df.columns if c.startswith(f"{role}_{prefix}")]
            feature_cols.extend(sorted(role_cols))
    if feature_type != "lantern_il_noratios":
        for role in ["IL", "HL", "CHL", "PEG"]:
            col = f"{role}_molratio"
            if col in encoded.df.columns and encoded.df[col].nunique() > 1:
                feature_cols.append(col)
        mr_col = "IL_to_nucleicacid_massratio"
        if mr_col in encoded.df.columns and encoded.df[mr_col].nunique() > 1:
            feature_cols.append("IL_to_nucleicacid_massratio")

    if context_features:
        from LNPBO.data.context import encode_context

        encoded.df, ctx_cols, _ = encode_context(encoded.df)
        feature_cols.extend(ctx_cols)
        print(f"  Context features ({len(ctx_cols)}): {ctx_cols[:5]}{'...' if len(ctx_cols) > 5 else ''}")

    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    valid_mask = encoded.df[feature_cols].notna().all(axis=1)
    encoded_df = encoded.df[valid_mask].copy()
    if "Formulation_ID" in encoded_df.columns:
        encoded_df = encoded_df.drop_duplicates(subset=["Formulation_ID"])
    encoded_df = encoded_df.reset_index(drop=True)
    encoded.df = encoded_df.copy()
    print(f"  Valid rows after cleanup: {len(encoded_df):,}")

    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(encoded_df))
    rng.shuffle(all_idx)

    seed_idx = sorted(all_idx[:n_seed])
    oracle_idx = sorted(all_idx[n_seed:])

    top_k_values = {
        10: set(encoded_df.nlargest(10, "Experiment_value").index),
        50: set(encoded_df.nlargest(50, "Experiment_value").index),
        100: set(encoded_df.nlargest(100, "Experiment_value").index),
    }

    print(f"  Seed pool: {len(seed_idx)} formulations")
    print(f"  Oracle pool: {len(oracle_idx)} formulations")
    print(f"  Top-10 Experiment_value threshold: {encoded_df['Experiment_value'].nlargest(10).min():.3f}")

    return encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values
