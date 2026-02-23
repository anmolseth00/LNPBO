from __future__ import annotations
import yaml
from sklearn.preprocessing import MinMaxScaler

def scale_formulation_df(df_to_scale_subset, df_to_scale_all, CONFIG_FILE):
    # Load configuration
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
    
    # Identify columns to scale
    to_scale = []
    for parameter in config['parameters']:
        if parameter['type'] != "MixtureRatiosParameter":
            to_scale.extend(parameter['columns'])
    
    # Initialize the scaler and fit on df_to_scale_all
    scaler = MinMaxScaler()
    scaler.fit(df_to_scale_all[to_scale])  # Use df_to_scale_all for determining min and max
    
    # Transform df_to_scale_subset using the fitted scaler
    df_to_scale_subset_scaled = df_to_scale_subset.copy()  # Avoid modifying the original DataFrame
    df_to_scale_subset_scaled[to_scale] = scaler.transform(df_to_scale_subset[to_scale])
    
    return df_to_scale_subset_scaled