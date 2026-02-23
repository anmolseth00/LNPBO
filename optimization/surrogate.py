from __future__ import annotations
from bayes_opt import BayesianOptimization, acquisition
from ..utils.ordering import order_df_columns
from ..space.parameters import ComponentParameter, DiscreteParameter, MixtureRatiosParameter
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml

def get_surrogate_model(
    formulation_screen_df_processed, 
    CONFIG_FILE, 
    round_number, 
    RANDOM_STATE_SEED=1
):
    """
    Retrieves the trained surrogate Gaussian Process model for deployment.
    
    Parameters:
    - formulation_screen_df_processed: DataFrame
        The processed formulation screen data.
    - CONFIG_FILE: str
        Path to the YAML configuration file.
    - round_number: int
        The round number used to define the random state.
    - RANDOM_STATE_SEED: int, optional (default=1)
        Base random state seed.
        
    Returns:
    - gp_model: The trained Gaussian Process surrogate model.
    """
    # Load configuration
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    # Copy and scale the DataFrame
    df = formulation_screen_df_processed.copy()
    to_scale = []
    for parameter in config['parameters']:
        if parameter['type'] != "MixtureRatiosParameter":
            to_scale.extend(parameter['columns'])
    scaler = MinMaxScaler()
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # Define columns and parameter bounds
    columns = []
    pbounds = {}
    PEG_MOLRATIO = 0.025  # Default for this dataset
    for parameter in config['parameters']:
        if parameter["type"] == "ComponentParameter":
            p_stringified = df[parameter["columns"]].astype(str).agg('_'.join, axis=1)
            encoded = pd.get_dummies(p_stringified)
            valid_options = np.stack(encoded.columns.str.split('_').to_series().apply(lambda x: [float(i) for i in x]).values)
            domain = np.stack([valid_options.min(axis=0), valid_options.max(axis=0)], axis=1)
            pbounds[parameter["name"]] = ComponentParameter(parameter["name"], domain, valid_options)
        elif parameter['type'] == "DiscreteParameter":
            pbounds[parameter['name']] = DiscreteParameter(parameter['name'], np.unique(df[parameter['columns'][0]]))
        elif parameter['type'] == "MixtureRatiosParameter":
            bounds = np.array([[df[c].min(), df[c].max()] for c in parameter['columns']])
            pbounds[parameter['name']] = MixtureRatiosParameter(
                parameter['name'], len(parameter['columns']), bounds=bounds, sum_to=(1 - PEG_MOLRATIO)
            )
    
    pbounds = {k: v for k, v in sorted(pbounds.items(), key=lambda x: x[0])}

    # Extract training data
    X_train, y_train = df[order_df_columns(config, pbounds)].values, df[config['target']].values

    # Set the random state for this round
    BATCH_SEED = RANDOM_STATE_SEED + round_number

    # Initialize Bayesian Optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=0,
        random_state=BATCH_SEED
    )

    optimizer.set_gp_params(alpha=1e-6, n_restarts_optimizer=20)

    # Register training data points
    for point, target in zip(X_train, y_train):
        optimizer.register(point, target)

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)

    # Return the trained Gaussian Process model
    return optimizer._gp