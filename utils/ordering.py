from __future__ import annotations


def order_df_columns(config, pbounds):
    res = []
    # pbounds are internally sorted, so this mirrors the behavior
    # of the optimizer
    for parameter in sorted(pbounds.keys()):
        for p in config["parameters"]:
            if p["name"] == parameter:
                res.extend(p["columns"])
    return res
