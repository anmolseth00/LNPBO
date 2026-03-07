import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler

from ..space.formulation import FormulationSpace
from ..space.parameters import ComponentParameter, DiscreteParameter, MixtureRatiosParameter
from ..utils.ordering import order_df_columns


def get_surrogate_model(
    data: pd.DataFrame,
    formulation_space: FormulationSpace,
    round_number: int,
    RANDOM_STATE_SEED: int = 1,
    ALPHA: float = 1e-6,
):
    """
    Train and return the GP surrogate model without running BO.

    Useful for analysis, visualization, and prediction on new data points.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with feature columns and Experiment_value.
    formulation_space : FormulationSpace
        Defines the search space.
    round_number : int
        Current optimization round.
    RANDOM_STATE_SEED : int
        Base random seed.
    ALPHA : float
        GP noise parameter.

    Returns
    -------
    dict
        Contains 'gp' (trained GP model), 'scaler' (fitted MinMaxScaler),
        'columns' (ordered feature columns), 'pbounds' (parameter dict).
    """
    config = formulation_space.get_configs()
    df = data.copy()

    # Collect all feature columns
    all_feature_cols = []
    for parameter in config["parameters"]:
        all_feature_cols.extend(parameter["columns"])

    # Drop rows with NaN features
    feature_cols_present = [c for c in all_feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols_present).reset_index(drop=True)

    # Scale non-mixture columns
    to_scale = []
    for parameter in config["parameters"]:
        if parameter["type"] != "MixtureRatiosParameter":
            to_scale.extend(parameter["columns"])

    scaler = MinMaxScaler()
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # Build parameter bounds
    columns = []
    pbounds = {}
    for parameter in config["parameters"]:
        if parameter["type"] == "ComponentParameter":
            p_stringified = df[parameter["columns"]].astype(str).agg("_".join, axis=1)
            encoded = pd.get_dummies(p_stringified)
            valid_options = np.stack(
                encoded.columns.str.split("_").to_series().apply(lambda x: [float(i) for i in x]).values
            )
            domain = np.stack([valid_options.min(axis=0), valid_options.max(axis=0)], axis=1)
            columns.append(parameter["columns"])
            pbounds[parameter["name"]] = ComponentParameter(parameter["name"], domain, valid_options)
        elif parameter["type"] == "DiscreteParameter":
            columns.append(parameter["columns"][0])
            pbounds[parameter["name"]] = DiscreteParameter(parameter["name"], np.unique(df[parameter["columns"][0]]))
        elif parameter["type"] == "MixtureRatiosParameter":
            columns_mixture = parameter["columns"]
            columns.append(parameter["columns"])
            bounds = np.array([[df[c].min(), df[c].max()] for c in parameter["columns"]])
            row_sums = df[columns_mixture].sum(axis=1)
            sum_to = row_sums.median()
            pbounds[parameter["name"]] = MixtureRatiosParameter(
                parameter["name"],
                len(columns_mixture),
                bounds=bounds,
                sum_to=sum_to,
            )

    pbounds = {k: v for k, v in sorted(pbounds.items(), key=lambda x: x[0])}
    columns = order_df_columns(config, dict(pbounds))

    BATCH_SEED = RANDOM_STATE_SEED + round_number
    X_train, y_train = df[columns].values, df[config["target"]].values

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=0,
        random_state=BATCH_SEED,
    )

    optimizer.set_gp_params(alpha=ALPHA, n_restarts_optimizer=20)

    for point, target in zip(X_train, y_train):
        optimizer.register(point, target)

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)

    return {
        "gp": optimizer._gp,
        "scaler": scaler,
        "columns": columns,
        "pbounds": pbounds,
        "X_train": X_train,
        "y_train": y_train,
    }
