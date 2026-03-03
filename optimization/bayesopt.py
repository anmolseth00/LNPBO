from __future__ import annotations
from bayes_opt import BayesianOptimization, acquisition
from .acquisition import KrigingBeliever, LogExpectedImprovement, LocalPenalization
from ..space.parameters import MixtureRatiosParameter, ComponentParameter, DiscreteParameter
from ..space.formulation import FormulationSpace
from .serialization import save_surrogate
from ..utils.ordering import order_df_columns

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def perform_bayesian_optimization(
    data: pd.DataFrame,
    formulation_space: FormulationSpace,
    round_number: int,
    acq_type: str,
    BATCH_SIZE: int = 25,
    RANDOM_STATE_SEED: int = 1,
    KAPPA: float = 5.0,
    XI: float = 0.01,
    ALPHA: float = 1e-6,
    verbose: int = 2,
    save_gp: bool = True,
):
    """
    Perform Bayesian Optimization over LNP formulation space.

    formulation_space fully replaces YAML configuration.
    """

    # Load configuration
    config = formulation_space.get_configs()

    df = data.copy()

    # Collect all feature columns used by parameters
    all_feature_cols = []
    for parameter in config['parameters']:
        all_feature_cols.extend(parameter['columns'])

    # Drop rows with NaN in any feature column (e.g., lipids without SMILES encodings)
    feature_cols_present = [c for c in all_feature_cols if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=feature_cols_present).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing feature values ({len(df)} remaining)")

    # Define and scale columns
    to_scale = []
    for parameter in config['parameters']:
        if parameter['type'] != "MixtureRatiosParameter":
            to_scale.extend(parameter['columns'])

    scaler = MinMaxScaler()
    df[to_scale] = scaler.fit_transform(df[to_scale])

    columns = []
    pbounds = {}
    for parameter in config['parameters']:
        if parameter["type"] == "ComponentParameter":
            p_stringified = df[parameter["columns"]].astype(str).agg('_'.join, axis=1)
            encoded = pd.get_dummies(p_stringified)
            valid_options = np.stack(encoded.columns.str.split('_').to_series().apply(lambda x: [float(i) for i in x]).values)
            domain = np.stack([valid_options.min(axis=0), valid_options.max(axis=0)], axis=1)
            columns.append(parameter["columns"])
            pbounds[parameter["name"]] = ComponentParameter(parameter["name"], domain, valid_options)
        elif parameter['type'] == "DiscreteParameter":
            if len(parameter['columns']) > 1:
                raise ValueError("DiscreteParameter with more than one column is not supported")
            columns.append(parameter['columns'][0])
            pbounds[parameter['name']] = DiscreteParameter(parameter['name'], np.unique(df[parameter['columns'][0]]))
        elif parameter['type'] == "MixtureRatiosParameter":
            columns_mixture = parameter['columns']
            columns.append(parameter["columns"])
            bounds = np.array([[df[c].min(), df[c].max()] for c in parameter['columns']])
            # Use median sum across all rows (robust to outliers)
            row_sums = df[columns_mixture].sum(axis=1)
            sum_to = row_sums.median()
            # Validate: warn if rows have inconsistent sums
            sum_std = row_sums.std()
            if sum_std > 0.01 * sum_to:
                print(f"Warning: molar ratio sums vary across rows (median={sum_to:.4f}, std={sum_std:.4f})")
            pbounds[parameter['name']] = MixtureRatiosParameter(
                parameter['name'],
                len(columns_mixture),
                bounds=bounds,
                sum_to=sum_to,
            )

    pbounds = {k: v for k, v in sorted(pbounds.items(), key=lambda x: x[0])}
    columns = order_df_columns(config, dict(pbounds))

    BATCH_SEED = RANDOM_STATE_SEED + round_number

    X_train, y_train = df[columns].values, df[config['target']].values

    # Acquisition function
    if acq_type == "UCB":
        acq = KrigingBeliever(
            acquisition.UpperConfidenceBound(
                kappa=KAPPA,
                random_state=BATCH_SEED,
            ),
            random_state=BATCH_SEED,
        )
    elif acq_type == "EI":
        acq = KrigingBeliever(
            acquisition.ExpectedImprovement(
                xi=XI,
                random_state=BATCH_SEED,
            ),
            random_state=BATCH_SEED,
        )
    elif acq_type == "LogEI":
        acq = KrigingBeliever(
            LogExpectedImprovement(
                xi=XI,
                random_state=BATCH_SEED,
            ),
            random_state=BATCH_SEED,
        )
    elif acq_type == "LP_UCB":
        acq = LocalPenalization(
            acquisition.UpperConfidenceBound(
                kappa=KAPPA,
                random_state=BATCH_SEED,
            ),
            random_state=BATCH_SEED,
        )
    elif acq_type == "LP_EI":
        acq = LocalPenalization(
            acquisition.ExpectedImprovement(
                xi=XI,
                random_state=BATCH_SEED,
            ),
            random_state=BATCH_SEED,
        )
    elif acq_type == "LP_LogEI":
        acq = LocalPenalization(
            LogExpectedImprovement(
                xi=XI,
                random_state=BATCH_SEED,
            ),
            random_state=BATCH_SEED,
        )
    else:
        raise ValueError(
            f"Unknown acq_type '{acq_type}'. "
            "Choose from: UCB, EI, LogEI, LP_UCB, LP_EI, LP_LogEI"
        )

    # Optimizer
    with warnings.catch_warnings():
        if verbose == 0:
            warnings.simplefilter("ignore")
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            acquisition_function=acq,
            verbose=verbose,
            random_state=BATCH_SEED,
            allow_duplicate_points=True,
        )

        optimizer.set_gp_params(alpha=ALPHA, n_restarts_optimizer=20)

        for x, y in zip(X_train, y_train):
            optimizer.register(x, y)

    if acq_type in ("EI", "LogEI"):
        acq.base_acquisition.y_max = np.max(y_train)
    elif acq_type in ("LP_EI", "LP_LogEI"):
        acq.base_acquisition.y_max = np.max(y_train)

    # Batch generation
    batch = []
    for _ in tqdm(range(BATCH_SIZE), desc="Selecting LNP formulations", disable=(verbose == 0)):
        point = optimizer.suggest()
        batch.append(
            dict(zip(columns, optimizer.space.params_to_array(point)))
        )

    # Save surrogate
    if save_gp:
        save_surrogate(
            f"round_{round_number}_gp.pkl",
            gp_model=optimizer._gp,
            scaler=scaler,
            columns=columns,
            metadata={"round": round_number},
        )

    # Post-processing (inverse scaling + fixed values)
    df_batch = pd.DataFrame(batch)
    df_batch[to_scale] = scaler.inverse_transform(df_batch[to_scale])

    # Re-insert fixed values (e.g. PEG molratio)
    for k, v in formulation_space.get_fixed_values().items():
        df_batch[k] = v

    return df_batch
