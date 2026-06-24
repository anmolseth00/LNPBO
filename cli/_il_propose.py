"""Generate novel ionizable lipid candidates via SELFIES mutation and scoring.

Uses SELFIES-based random mutation of known ILs, filters by
physicochemical constraints, scores with a conformal XGBoost model,
and selects a diverse output set via MaxMin picking on Tanimoto distance.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from ..data.compute_pcs import compute_pcs

_TERTIARY_AMINE_SMARTS = "[NX3;H0;!$(N-C=O);!$([nR])]"


@dataclass
class EncoderBundle:
    """Holds a fitted dimensionality reducer and scaler for one feature type.

    Attributes:
        reducer: Fitted PCA/PLS reducer object.
        scaler: Fitted StandardScaler (or similar) for pre-reduction scaling.
        n_components: Number of principal components retained.
    """
    reducer: object
    scaler: object
    n_components: int


@dataclass
class LanternEncoders:
    """Container for the pair of LANTERN encoder bundles (count MFP + RDKit).

    Attributes:
        count_mfp: Encoder bundle for count Morgan fingerprints, or ``None``.
        rdkit: Encoder bundle for RDKit descriptors, or ``None``.
    """

    count_mfp: EncoderBundle | None
    rdkit: EncoderBundle | None


def _canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or ``None`` if parsing fails.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string, or ``None`` for invalid input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_il_training_table(path: str) -> pd.DataFrame:
    """Load and deduplicate an IL training table from CSV.

    Reads columns ``IL_name``, ``IL_SMILES``, and ``Experiment_value``,
    canonicalizes SMILES, drops invalid entries, and averages duplicate
    (name, SMILES) pairs.

    Args:
        path: Path to a CSV file with at least the three required columns.

    Returns:
        DataFrame with columns ``IL_name``, ``IL_SMILES``, and
        ``Experiment_value`` (one row per unique IL).

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    required = {"IL_name", "IL_SMILES", "Experiment_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for IL training: {sorted(missing)}")

    df = df[list(required)].dropna()
    df["IL_SMILES"] = df["IL_SMILES"].map(_canonicalize_smiles)
    df = df.dropna(subset=["IL_SMILES"])

    grouped = (
        df.groupby(["IL_name", "IL_SMILES"], as_index=False)["Experiment_value"]
        .mean()
    )
    return grouped


def _fit_lantern_encoders(
    smiles: list[str],
    targets: np.ndarray,
    n_pcs_count_mfp: int,
    n_pcs_rdkit: int,
    reduction: str,
) -> tuple[np.ndarray, LanternEncoders]:
    """Fit LANTERN encoders on training SMILES and return feature matrix.

    Computes count Morgan FP and/or RDKit descriptor PCs via
    ``compute_pcs``, stores the fitted reducers/scalers in a
    ``LanternEncoders`` bundle, and horizontally stacks the resulting
    feature blocks.

    Args:
        smiles: Training SMILES strings.
        targets: Corresponding experiment values (used by PLS reduction).
        n_pcs_count_mfp: Number of count Morgan FP PCs (0 to skip).
        n_pcs_rdkit: Number of RDKit descriptor PCs (0 to skip).
        reduction: Reduction method (``"pca"``, ``"pls"``, or ``"none"``).

    Returns:
        Tuple of (feature matrix of shape ``(len(smiles), total_pcs)``,
        fitted ``LanternEncoders``).

    Raises:
        ValueError: If both PC counts are 0.
    """
    blocks = []
    count_bundle = None
    rdkit_bundle = None

    if n_pcs_count_mfp > 0:
        pcs, reducer, scaler, _ = compute_pcs(
            smiles,
            feature_type="count_mfp",
            experiment_values=targets.tolist(),
            n_components=n_pcs_count_mfp,
            reduction=reduction,
        )
        blocks.append(pcs)
        count_bundle = EncoderBundle(reducer=reducer, scaler=scaler, n_components=pcs.shape[1])

    if n_pcs_rdkit > 0:
        pcs, reducer, scaler, _ = compute_pcs(
            smiles,
            feature_type="rdkit",
            experiment_values=targets.tolist(),
            n_components=n_pcs_rdkit,
            reduction=reduction,
        )
        blocks.append(pcs)
        rdkit_bundle = EncoderBundle(reducer=reducer, scaler=scaler, n_components=pcs.shape[1])

    if not blocks:
        raise ValueError("At least one of count_mfp or rdkit PCs must be > 0.")

    features = np.hstack(blocks)
    return features, LanternEncoders(count_mfp=count_bundle, rdkit=rdkit_bundle)


def _encode_lantern(
    smiles: list[str],
    encoders: LanternEncoders,
    reduction: str,
) -> np.ndarray:
    """Encode SMILES into LANTERN features using pre-fitted encoders.

    Args:
        smiles: SMILES strings to encode.
        encoders: Pre-fitted ``LanternEncoders`` from ``_fit_lantern_encoders``.
        reduction: Reduction method (must match the one used during fitting).

    Returns:
        Feature matrix of shape ``(len(smiles), total_pcs)``.

    Raises:
        ValueError: If no encoders are available.
    """
    blocks = []
    if encoders.count_mfp is not None:
        pcs, _, _, _ = compute_pcs(
            smiles,
            feature_type="count_mfp",
            n_components=encoders.count_mfp.n_components,
            reduction=reduction,
            fitted_reducer=encoders.count_mfp.reducer,
            fitted_scaler=encoders.count_mfp.scaler,
        )
        blocks.append(pcs)
    if encoders.rdkit is not None:
        pcs, _, _, _ = compute_pcs(
            smiles,
            feature_type="rdkit",
            n_components=encoders.rdkit.n_components,
            reduction=reduction,
            fitted_reducer=encoders.rdkit.reducer,
            fitted_scaler=encoders.rdkit.scaler,
        )
        blocks.append(pcs)
    if not blocks:
        raise ValueError("No encoders available for LANTERN features.")
    return np.hstack(blocks)


def _build_selfies_token_set(smiles_list: Iterable[str]) -> list[str]:
    """Extract the sorted set of unique SELFIES tokens from a SMILES corpus.

    Args:
        smiles_list: Iterable of SMILES strings.

    Returns:
        Sorted list of unique SELFIES token strings.
    """
    tokens = set()
    for smi in smiles_list:
        try:
            s = sf.encoder(smi)
            for tok in sf.split_selfies(s):
                tokens.add(tok)
        except sf.EncoderError:
            continue
    return sorted(tokens)


def _mutate_selfies(
    seed_smiles: str,
    token_set: list[str],
    rng: np.random.Generator,
    n_mutations: int,
    max_tokens: int | None,
) -> str | None:
    """Apply random SELFIES mutations to a seed molecule.

    Performs ``n_mutations`` random operations (substitution, insertion,
    or deletion) on the SELFIES token sequence of ``seed_smiles``.

    Args:
        seed_smiles: SMILES string of the seed molecule.
        token_set: Vocabulary of allowed SELFIES tokens.
        rng: NumPy random generator.
        n_mutations: Number of mutation operations to apply.
        max_tokens: Maximum token-sequence length (insertions beyond this
            are skipped), or ``None`` for no limit.

    Returns:
        Canonical SMILES of the mutated molecule, or ``None`` if encoding,
        decoding, or canonicalization fails.
    """
    try:
        seed_selfies = sf.encoder(seed_smiles)
    except sf.EncoderError:
        return None

    tokens = list(sf.split_selfies(seed_selfies))
    if not tokens:
        return None

    for _ in range(n_mutations):
        op = rng.choice(["sub", "ins", "del"])
        if op == "del" and len(tokens) <= 1:
            op = rng.choice(["sub", "ins"])

        if op == "sub":
            idx = rng.integers(0, len(tokens))
            tokens[idx] = rng.choice(token_set)
        elif op == "ins":
            if max_tokens is not None and len(tokens) >= max_tokens:
                continue
            idx = rng.integers(0, len(tokens) + 1)
            tokens.insert(idx, rng.choice(token_set))
        elif op == "del":
            idx = rng.integers(0, len(tokens))
            tokens.pop(idx)

    mutated_selfies = "".join(tokens)
    try:
        smiles = sf.decoder(mutated_selfies)
    except sf.DecoderError:
        return None

    return _canonicalize_smiles(smiles)


def _has_tertiary_amine(mol: Chem.Mol, patt: Chem.Mol) -> bool:
    """Check whether a molecule contains a tertiary amine substructure.

    Args:
        mol: RDKit molecule object.
        patt: Compiled SMARTS pattern for tertiary amine matching.

    Returns:
        ``True`` if the pattern is found in ``mol``.
    """
    return mol.HasSubstructMatch(patt)


def _filter_smiles(
    smiles: str,
    amine_patt: Chem.Mol,
    mw_min: float | None,
    mw_max: float | None,
    logp_min: float | None,
    logp_max: float | None,
    max_atoms: int | None,
) -> tuple[bool, dict[str, float]]:
    """Apply physicochemical filters to a candidate SMILES.

    Checks for valid parsing, tertiary amine presence, atom count,
    molecular weight, and logP.

    Args:
        smiles: Candidate SMILES string.
        amine_patt: Compiled SMARTS pattern for tertiary amine.
        mw_min: Minimum molecular weight, or ``None`` to skip.
        mw_max: Maximum molecular weight, or ``None`` to skip.
        logp_min: Minimum Crippen logP, or ``None`` to skip.
        logp_max: Maximum Crippen logP, or ``None`` to skip.
        max_atoms: Maximum heavy-atom count, or ``None`` to skip.

    Returns:
        Tuple of (passed: bool, properties: dict).  ``properties`` contains
        ``"mw"`` and/or ``"logp"`` if those filters were evaluated.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {}

    if not _has_tertiary_amine(mol, amine_patt):
        return False, {}

    if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
        return False, {}

    props = {}
    if mw_min is not None or mw_max is not None:
        mw = Descriptors.MolWt(mol)
        props["mw"] = mw
        if mw_min is not None and mw < mw_min:
            return False, {}
        if mw_max is not None and mw > mw_max:
            return False, {}
    if logp_min is not None or logp_max is not None:
        logp = Descriptors.MolLogP(mol)
        props["logp"] = logp
        if logp_min is not None and logp < logp_min:
            return False, {}
        if logp_max is not None and logp > logp_max:
            return False, {}

    return True, props


def _train_uncertainty_model(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: int,
    n_jobs: int,
    confidence_level: float,
) -> tuple[object, object | None]:
    """Train an XGBoost model with MAPIE cross-conformal uncertainty.

    If the training set has fewer than 2 samples, trains a plain XGBoost
    regressor without conformal calibration.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target values of shape ``(n_samples,)``.
        random_seed: Random seed for reproducibility.
        n_jobs: Number of parallel jobs.
        confidence_level: Confidence level for conformal prediction
            intervals (e.g. 0.68 for ~1-sigma).

    Returns:
        Tuple of (base XGBRegressor, fitted MAPIE regressor or ``None``).
    """
    from mapie.regression import CrossConformalRegressor
    from xgboost import XGBRegressor

    if len(X) < 2:
        model = XGBRegressor(n_estimators=200, random_state=random_seed, n_jobs=n_jobs, verbosity=0)
        model.fit(X, y)
        return model, None

    n_cv = min(5, len(X))
    if n_cv < 2:
        n_cv = 2

    base = XGBRegressor(n_estimators=200, random_state=random_seed, n_jobs=n_jobs, verbosity=0)
    mapie = CrossConformalRegressor(
        base,
        method="plus",
        cv=n_cv,
        confidence_level=confidence_level,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    mapie.fit_conformalize(X, y)
    return base, mapie


def _predict_with_uncertainty(
    model: object,
    mapie: object | None,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict with optional conformal uncertainty estimates.

    If ``mapie`` is ``None``, returns zero uncertainty.

    Args:
        model: Fitted base model (used as fallback when ``mapie`` is
            ``None``).
        mapie: Fitted MAPIE ``CrossConformalRegressor``, or ``None``.
        X: Feature matrix of shape ``(n_samples, n_features)``.

    Returns:
        Tuple of (mean predictions, half-width of interval, lower bound,
        upper bound), each of shape ``(n_samples,)``.
    """
    if mapie is None:
        preds = model.predict(X)
        return preds, np.zeros(len(preds)), preds, preds

    y_pred, y_intervals = mapie.predict_interval(X)
    lower = y_intervals[:, 0, 0]
    upper = y_intervals[:, 1, 0]
    half_width = (upper - lower) / 2
    return y_pred, half_width, lower, upper


def _fingerprint_smiles(smiles: str, generator):
    """Compute a molecular fingerprint from a SMILES string.

    Args:
        smiles: SMILES string.
        generator: RDKit fingerprint generator (e.g. Morgan generator).

    Returns:
        RDKit fingerprint object, or ``None`` if SMILES parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)


def _maxmin_select(
    fps: list,
    scores: np.ndarray,
    n_select: int,
) -> list[int]:
    """Select a diverse subset using MaxMin picking on Tanimoto distance.

    Starts from the highest-scoring candidate, then iteratively picks the
    candidate most distant (in Tanimoto space) from all previously
    selected ones.

    Args:
        fps: List of RDKit fingerprint objects.
        scores: Array of acquisition scores (used to pick the seed).
        n_select: Number of candidates to select.

    Returns:
        List of integer indices into ``fps``/``scores`` for the selected
        candidates.
    """
    if n_select >= len(fps):
        return list(range(len(fps)))

    start_idx = int(np.argmax(scores))
    selected = [start_idx]

    sims = DataStructs.BulkTanimotoSimilarity(fps[start_idx], fps)
    min_dist = 1.0 - np.array(sims)
    min_dist[start_idx] = -1.0

    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        sims = DataStructs.BulkTanimotoSimilarity(fps[next_idx], fps)
        dist = 1.0 - np.array(sims)
        min_dist = np.minimum(min_dist, dist)
        for idx in selected:
            min_dist[idx] = -1.0

    return selected


def propose_ionizable_lipids(
    dataset_path: str,
    n_candidates: int = 20000,
    n_output: int = 100,
    diversity_pool: int = 1000,
    random_seed: int = 42,
    max_mutations: int = 2,
    lcb_kappa: float = 1.0,
    lcb_mode: str = "std",
    n_jobs: int = 1,
    confidence_level: float = 0.68,
    n_pcs_count_mfp: int = 5,
    n_pcs_rdkit: int = 5,
    reduction: str = "pls",
    amine_smarts: str = _TERTIARY_AMINE_SMARTS,
    mw_min: float | None = None,
    mw_max: float | None = None,
    logp_min: float | None = None,
    logp_max: float | None = None,
    max_atoms: int | None = None,
    max_attempts: int | None = None,
) -> pd.DataFrame:
    """Generate and rank novel ionizable lipid candidates.

    Pipeline: (1) load IL training data, (2) fit LANTERN encoders and a
    conformal XGBoost model, (3) generate candidates via SELFIES mutation
    with physicochemical filtering, (4) score by lower confidence bound,
    (5) select a diverse final set via MaxMin Tanimoto picking, and
    (6) annotate each candidate with its nearest known IL.

    Args:
        dataset_path: Path to CSV with ``IL_name``, ``IL_SMILES``, and
            ``Experiment_value`` columns.
        n_candidates: Total number of valid mutants to generate before
            scoring.
        n_output: Number of candidates in the final diverse output set.
        diversity_pool: Size of the top-scoring pool from which MaxMin
            picking selects ``n_output`` diverse candidates.
        random_seed: Random seed for all stochastic operations.
        max_mutations: Maximum number of SELFIES mutations per candidate.
        lcb_kappa: Exploration weight for ``mean - kappa * std`` scoring
            (only used when ``lcb_mode="std"``).
        lcb_mode: Scoring mode -- ``"std"`` for ``mean - kappa * std``,
            ``"lower"`` for the conformal lower bound directly.
        n_jobs: Number of parallel jobs for XGBoost and MAPIE.
        confidence_level: Confidence level for conformal intervals.
        n_pcs_count_mfp: Number of count Morgan FP PCs for LANTERN.
        n_pcs_rdkit: Number of RDKit descriptor PCs for LANTERN.
        reduction: Dimensionality reduction method (``"pca"``, ``"pls"``,
            or ``"none"``).
        amine_smarts: SMARTS pattern for required tertiary amine.
        mw_min: Minimum molecular weight filter, or ``None``.
        mw_max: Maximum molecular weight filter, or ``None``.
        logp_min: Minimum logP filter, or ``None``.
        logp_max: Maximum logP filter, or ``None``.
        max_atoms: Maximum heavy-atom count filter, or ``None``.
        max_attempts: Maximum mutation attempts before stopping. Defaults
            to ``max(10000, n_candidates * 50)``.

    Returns:
        DataFrame with columns including ``candidate_smiles``,
        ``seed_smiles``, ``pred_mean``, ``pred_std``, ``lcb_score``,
        ``nearest_il_name``, ``nearest_il_smiles``, and
        ``nearest_il_similarity``, sorted by descending ``lcb_score``.

    Raises:
        ValueError: If ``confidence_level`` is out of range,
            ``lcb_mode`` is invalid, ``amine_smarts`` cannot be parsed,
            or no valid candidates are generated.
    """
    rng = np.random.default_rng(random_seed)
    if not (0.0 < confidence_level <= 1.0):
        raise ValueError("confidence_level must be in (0, 1].")
    if lcb_mode not in {"std", "lower"}:
        raise ValueError("lcb_mode must be 'std' or 'lower'.")
    amine_patt = Chem.MolFromSmarts(amine_smarts)
    if amine_patt is None:
        raise ValueError(f"Invalid SMARTS pattern: {amine_smarts!r}")

    train_df = load_il_training_table(dataset_path)
    train_smiles = train_df["IL_SMILES"].tolist()
    train_targets = train_df["Experiment_value"].to_numpy(dtype=float)

    X_train, encoders = _fit_lantern_encoders(
        train_smiles,
        train_targets,
        n_pcs_count_mfp=n_pcs_count_mfp,
        n_pcs_rdkit=n_pcs_rdkit,
        reduction=reduction,
    )

    model, mapie = _train_uncertainty_model(
        X_train,
        train_targets,
        random_seed,
        n_jobs=n_jobs,
        confidence_level=confidence_level,
    )

    token_set = _build_selfies_token_set(train_smiles)
    if not token_set:
        raise ValueError("Failed to build SELFIES token set from training ILs.")

    existing = set(train_smiles)
    generated = []
    attempts = 0
    max_attempts = max_attempts or max(10000, n_candidates * 50)

    while len(generated) < n_candidates and attempts < max_attempts:
        attempts += 1
        seed = rng.choice(train_smiles)
        n_mut = rng.integers(1, max_mutations + 1)
        cand = _mutate_selfies(seed, token_set, rng, n_mutations=int(n_mut), max_tokens=None)
        if cand is None or cand in existing:
            continue

        ok, props = _filter_smiles(
            cand,
            amine_patt=amine_patt,
            mw_min=mw_min,
            mw_max=mw_max,
            logp_min=logp_min,
            logp_max=logp_max,
            max_atoms=max_atoms,
        )
        if not ok:
            continue

        existing.add(cand)
        generated.append(
            {
                "candidate_smiles": cand,
                "seed_smiles": seed,
                **props,
            }
        )

    if not generated:
        raise ValueError("No valid candidates generated. Try increasing max_attempts or relaxing filters.")

    gen_df = pd.DataFrame(generated)

    X_cand = _encode_lantern(gen_df["candidate_smiles"].tolist(), encoders, reduction=reduction)
    pred_mean, pred_std, pred_lower, pred_upper = _predict_with_uncertainty(model, mapie, X_cand)

    gen_df["pred_mean"] = pred_mean
    gen_df["pred_std"] = pred_std
    gen_df["pred_lower"] = pred_lower
    gen_df["pred_upper"] = pred_upper
    if lcb_mode == "lower":
        gen_df["lcb_score"] = pred_lower
    else:
        gen_df["lcb_score"] = pred_mean - lcb_kappa * pred_std

    order = np.argsort(gen_df["lcb_score"].to_numpy())[::-1]
    pool_size = min(diversity_pool, len(order))
    pool_idx = order[:pool_size]

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    pool_fps = []
    pool_rows = []
    for idx in pool_idx:
        smi = gen_df.iloc[idx]["candidate_smiles"]
        fp = _fingerprint_smiles(smi, fp_gen)
        if fp is None:
            continue
        pool_fps.append(fp)
        pool_rows.append(idx)

    if not pool_rows:
        raise ValueError("No valid fingerprints generated for candidate pool.")

    pool_scores = gen_df.loc[pool_rows, "lcb_score"].to_numpy()
    selected_local = _maxmin_select(pool_fps, pool_scores, n_select=min(n_output, len(pool_rows)))
    selected_idx = [pool_rows[i] for i in selected_local]

    selected_df = gen_df.loc[selected_idx].copy()

    # Nearest-neighbor IL for chemist anchoring
    train_valid = []
    for _, row in train_df.iterrows():
        fp = _fingerprint_smiles(row["IL_SMILES"], fp_gen)
        if fp is None:
            continue
        train_valid.append((row["IL_name"], row["IL_SMILES"], fp))
    train_fps = [t[2] for t in train_valid]
    if not train_valid:
        raise ValueError("No valid IL fingerprints found in training data.")

    nn_names = []
    nn_smiles = []
    nn_sims = []
    for smi in selected_df["candidate_smiles"].tolist():
        fp = _fingerprint_smiles(smi, fp_gen)
        if fp is None:
            nn_names.append(None)
            nn_smiles.append(None)
            nn_sims.append(np.nan)
            continue

        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        best_idx = int(np.argmax(sims))
        nn_names.append(train_valid[best_idx][0])
        nn_smiles.append(train_valid[best_idx][1])
        nn_sims.append(float(sims[best_idx]))

    selected_df["nearest_il_name"] = nn_names
    selected_df["nearest_il_smiles"] = nn_smiles
    selected_df["nearest_il_similarity"] = nn_sims

    selected_df = selected_df.sort_values("lcb_score", ascending=False).reset_index(drop=True)
    return selected_df
