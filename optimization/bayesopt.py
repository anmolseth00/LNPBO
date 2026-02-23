from __future__ import annotations
from bayes_opt import BayesianOptimization, acquisition
from .acquisition import KrigingBeliever
from ..space.parameters import MixtureRatiosParameter, ComponentParameter, DiscreteParameter
from ..space.formulation import FormulationSpace
from .serialization import save_surrogate
from ..utils.ordering import order_df_columns

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
):
    """
    Perform Bayesian Optimization over LNP formulation space.

    formulation_space fully replaces YAML configuration.
    """

    # Load configuration
    config = formulation_space.get_configs()

    df = data.copy()

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
            pbounds[parameter['name']] = MixtureRatiosParameter(
                parameter['name'],
                len(columns_mixture),
                bounds=bounds,
                sum_to=df[columns_mixture].iloc[0].sum() #(1)
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
    else:
        raise ValueError("Acquisition function type must be 'UCB' or 'EI'")

    # Optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        acquisition_function=acq,
        verbose=2,
        random_state=BATCH_SEED,
        allow_duplicate_points=False,
    )

    optimizer.set_gp_params(alpha=ALPHA, n_restarts_optimizer=20)

    for x, y in zip(X_train, y_train):
        optimizer.register(x, y)

    if acq_type == "EI":
        acq.base_acquisition.y_max = np.max(y_train)

    # Batch generation
    batch = []
    for _ in tqdm(range(BATCH_SIZE), desc="Selecting LNP formulations"):
        point = optimizer.suggest()
        batch.append(
            dict(zip(columns, optimizer.space.params_to_array(point)))
        )

    # Save surrogate
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
