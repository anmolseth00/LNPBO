#!/usr/bin/env python3
"""
SHAP Analysis: Feature Attribution for LNPBO XGBoost Surrogate
==============================================================

Trains XGBoost on the full LNPDB with LANTERN features + experimental context,
then uses SHAP (TreeExplainer) to quantify feature importance. This answers:

1. How much signal comes from molecular features vs experimental context?
2. Which specific context variables (cell type, target, RoA, etc.) matter most?
3. Are we learning "this molecule is good" or "this assay gives high numbers"?

Usage:
    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --no-context   # molecular features only
    python scripts/shap_analysis.py --subset 5000   # faster iteration
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from LNPBO.data.context import encode_context
from LNPBO.data.dataset import Dataset, encoders_for_feature_type
from LNPBO.data.lnpdb_bridge import load_lnpdb_full
from LNPBO.models.splits import scaffold_split
from LNPBO.optimization.optimizer import ENC_PREFIXES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_features(df, encoded_df, feature_type="lantern", reduction="pls", context_features=True):
    """Build feature matrix with column names for SHAP."""
    feature_cols = []
    for role in ["IL", "HL", "CHL", "PEG"]:
        for prefix in ENC_PREFIXES:
            role_cols = [c for c in encoded_df.columns if c.startswith(f"{role}_{prefix}")]
            feature_cols.extend(sorted(role_cols))
    for role in ["IL", "HL", "CHL", "PEG"]:
        col = f"{role}_molratio"
        if col in encoded_df.columns and encoded_df[col].nunique() > 1:
            feature_cols.append(col)
    if "IL_to_nucleicacid_massratio" in encoded_df.columns and encoded_df["IL_to_nucleicacid_massratio"].nunique() > 1:
        feature_cols.append("IL_to_nucleicacid_massratio")

    ctx_cols = []
    ctx_levels = None
    if context_features:
        encoded_df, ctx_cols, ctx_levels = encode_context(encoded_df)
        feature_cols.extend(ctx_cols)

    return encoded_df, feature_cols, ctx_cols, ctx_levels


def categorize_features(feature_cols):
    """Group feature columns into semantic categories for aggregated SHAP."""
    categories = {}
    for col in feature_cols:
        if col.startswith("ctx_"):
            # e.g. ctx_Model_type__HeLa -> Model_type
            parts = col.split("__")
            cat = parts[0].replace("ctx_", "")
            categories[col] = f"Context: {cat}"
        elif "_molratio" in col:
            categories[col] = "Molar ratios"
        elif "massratio" in col:
            categories[col] = "Mass ratio"
        else:
            # e.g. IL_count_mfp_pc1 -> IL fingerprint
            role = col.split("_")[0]
            categories[col] = f"{role} fingerprint"
    return categories


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis of LNPBO XGBoost surrogate")
    parser.add_argument("--subset", type=int, default=None, help="Use N-row subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-context", action="store_true", help="Exclude context features")
    parser.add_argument("--feature-type", type=str, default="lantern", choices=["mfp", "lantern", "count_mfp", "rdkit"])
    parser.add_argument("--reduction", type=str, default="pls", choices=["pca", "pls", "none"])
    parser.add_argument("--output-dir", type=str, default="shap_output")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument(
        "--split",
        type=str,
        default="amine",
        choices=["amine", "scaffold", "random"],
        help="Split strategy (default: amine on IL_head_name, matching LNPDB paper)",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)
    use_context = not args.no_context

    # -------------------------------------------------------------------------
    # 1. Load and encode data
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)
    print(f"Context features: {use_context}")
    print(f"Feature type: {args.feature_type}, reduction: {args.reduction}")
    print(f"Split strategy: {args.split}")

    t0 = time.time()
    dataset = load_lnpdb_full()
    df = dataset.df

    if args.subset and args.subset < len(df):
        df = df.sample(n=args.subset, random_state=args.seed).reset_index(drop=True)
        df["Formulation_ID"] = range(1, len(df) + 1)
        dataset = Dataset(df, source="lnpdb", name="LNPDB_shap")

    print(f"Loaded {len(df):,} formulations ({time.time() - t0:.1f}s)")

    # Encode molecular features
    def _should_encode(role, default_n=5):
        name_col = f"{role}_name"
        smiles_col = f"{role}_SMILES"
        if df[name_col].nunique() <= 1:
            return 0
        if smiles_col not in df.columns or df[smiles_col].dropna().nunique() <= 1:
            return 0
        return default_n

    il_pcs = _should_encode("IL")
    hl_pcs = _should_encode("HL", 3)
    chl_pcs = _should_encode("CHL", 3)
    peg_pcs = _should_encode("PEG", 3)

    enc = encoders_for_feature_type(args.feature_type, il_pcs=il_pcs, other_pcs=0)
    for role, n in [("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
        if role in enc:
            enc[role] = {k: n for k in enc[role]}

    encoded = dataset.encode_dataset(enc, reduction=args.reduction)
    encoded_df = encoded.df.copy()

    # Drop rows with NaN in target
    encoded_df = encoded_df.dropna(subset=["Experiment_value"]).reset_index(drop=True)

    # Build feature matrix
    encoded_df, feature_cols, ctx_cols, _ = build_features(
        df,
        encoded_df,
        feature_type=args.feature_type,
        reduction=args.reduction,
        context_features=use_context,
    )

    # Drop rows with NaN features
    valid_mask = encoded_df[feature_cols].notna().all(axis=1)
    encoded_df = encoded_df[valid_mask].reset_index(drop=True)
    print(f"Valid rows: {len(encoded_df):,}, features: {len(feature_cols)}")
    print(f"  Molecular features: {len(feature_cols) - len(ctx_cols)}")
    print(f"  Context features: {len(ctx_cols)}")

    # -------------------------------------------------------------------------
    # 2. Train/test split
    # -------------------------------------------------------------------------
    X = encoded_df[feature_cols].values
    y = encoded_df["Experiment_value"].values
    feature_names = feature_cols

    if args.split == "amine":
        head_col = "IL_head_name"
        if head_col not in encoded_df.columns:
            print(f"Warning: {head_col} not found, falling back to random split")
            args.split = "random"
        else:
            groups = encoded_df[head_col].fillna("Unknown")
            unique_heads = groups.unique()
            rng = np.random.RandomState(args.seed)
            rng.shuffle(unique_heads)
            n_test_heads = max(1, int(len(unique_heads) * args.test_size))
            test_heads = set(unique_heads[:n_test_heads])
            test_mask = groups.isin(test_heads)
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print(f"Amine split — {len(unique_heads)} groups, {len(test_heads)} held out")
            print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    if args.split == "scaffold":
        il_smiles = encoded_df["IL_SMILES"].tolist()
        train_idx, val_idx, test_idx = scaffold_split(
            il_smiles,
            sizes=(1.0 - args.test_size, 0.0, args.test_size),
            seed=args.seed,
        )
        train_idx = train_idx + val_idx
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Scaffold split — Train: {len(X_train):,}, Test: {len(X_test):,}")
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
        )
        print(f"Random split — Train: {len(X_train):,}, Test: {len(X_test):,}")

    # -------------------------------------------------------------------------
    # 3. Train XGBoost
    # -------------------------------------------------------------------------
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        colsample_bytree=0.7,
        subsample=0.8,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=0,
    )
    print("\nTraining XGBoost...")
    t0 = time.time()
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    elapsed = time.time() - t0

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = root_mean_squared_error(y_test, y_pred_test)

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  ({elapsed:.1f}s)")

    # -------------------------------------------------------------------------
    # 4. SHAP analysis
    # -------------------------------------------------------------------------
    import shap

    print("\nComputing SHAP values (TreeExplainer)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)

    # Use test set for SHAP (representative of unseen data)
    shap_values = explainer.shap_values(X_test_s)
    print(f"  SHAP computed on {len(X_test_s):,} test samples ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------------------
    # 5. Aggregate SHAP by feature category
    # -------------------------------------------------------------------------
    categories = categorize_features(feature_names)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "category": [categories[f] for f in feature_names],
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    # Aggregate by category
    cat_importance = feature_importance.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=False)
    total_shap = cat_importance.sum()
    cat_pct = (cat_importance / total_shap * 100).round(1)

    print("\n" + "=" * 70)
    print("SHAP ATTRIBUTION BY CATEGORY")
    print("=" * 70)
    for cat in cat_importance.index:
        print(f"  {cat:<30} {cat_importance[cat]:>8.4f}  ({cat_pct[cat]:>5.1f}%)")

    # Context vs molecular split
    ctx_shap = cat_importance[[c for c in cat_importance.index if c.startswith("Context")]].sum()
    mol_shap = total_shap - ctx_shap
    print(f"\n  Molecular total:  {mol_shap:.4f} ({mol_shap / total_shap * 100:.1f}%)")
    print(f"  Context total:    {ctx_shap:.4f} ({ctx_shap / total_shap * 100:.1f}%)")

    # Top 20 individual features
    print("\nTop 20 features:")
    for _, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:<45} {row['mean_abs_shap']:>8.4f}  [{row['category']}]")

    # -------------------------------------------------------------------------
    # 6. Save results and plots
    # -------------------------------------------------------------------------
    results = {
        "config": {
            "feature_type": args.feature_type,
            "reduction": args.reduction,
            "context_features": use_context,
            "subset": args.subset,
            "seed": args.seed,
            "split": args.split,
            "test_size": args.test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "n_context_features": len(ctx_cols),
        },
        "model_performance": {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "n_estimators": int(model.best_iteration + 1) if hasattr(model, "best_iteration") else 500,
        },
        "category_importance": {
            cat: {"shap": float(cat_importance[cat]), "pct": float(cat_pct[cat])} for cat in cat_importance.index
        },
        "molecular_vs_context": {
            "molecular_shap": float(mol_shap),
            "molecular_pct": float(mol_shap / total_shap * 100),
            "context_shap": float(ctx_shap),
            "context_pct": float(ctx_shap / total_shap * 100),
        },
        "top_20_features": [
            {"feature": row["feature"], "shap": float(row["mean_abs_shap"]), "category": row["category"]}
            for _, row in feature_importance.head(20).iterrows()
        ],
    }

    suffix = "with_context" if use_context else "no_context"
    json_path = output_dir / f"shap_results_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Save full feature importance table
    csv_path = output_dir / f"shap_feature_importance_{suffix}.csv"
    feature_importance.to_csv(csv_path, index=False)
    print(f"Feature importance saved to {csv_path}")

    # Generate plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Category bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = []
        for cat in cat_importance.index:
            if cat.startswith("Context"):
                colors.append("#e74c3c")
            elif "fingerprint" in cat:
                colors.append("#3498db")
            else:
                colors.append("#2ecc71")
        ax.barh(range(len(cat_importance)), cat_importance.values, color=colors)
        ax.set_yticks(range(len(cat_importance)))
        ax.set_yticklabels(cat_importance.index, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Attribution by Category (Test R²={test_r2:.3f})")
        ax.invert_yaxis()
        for i, (val, pct) in enumerate(zip(cat_importance.values, cat_pct.values)):
            ax.text(val + total_shap * 0.01, i, f"{pct:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"shap_categories_{suffix}.png", dpi=300, bbox_inches="tight")
        print(f"Category plot saved to {output_dir / f'shap_categories_{suffix}.png'}")

        # Plot 2: SHAP beeswarm (top 30 features)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test_s,
            feature_names=feature_names,
            max_display=30,
            show=False,
        )
        plt.title(f"SHAP Summary (Top 30 Features, Test R²={test_r2:.3f})")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_beeswarm_{suffix}.png", dpi=300, bbox_inches="tight")
        print(f"Beeswarm plot saved to {output_dir / f'shap_beeswarm_{suffix}.png'}")

        # Plot 3: Molecular vs Context pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            [mol_shap, ctx_shap],
            labels=[
                f"Molecular\n({mol_shap / total_shap * 100:.1f}%)",
                f"Context\n({ctx_shap / total_shap * 100:.1f}%)",
            ],
            colors=["#3498db", "#e74c3c"],
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax.set_title(f"Signal Source: Molecular vs Experimental Context\n(Test R²={test_r2:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / f"shap_pie_{suffix}.png", dpi=300, bbox_inches="tight")
        print(f"Pie chart saved to {output_dir / f'shap_pie_{suffix}.png'}")

        plt.close("all")

    except ImportError:
        print("matplotlib or shap plot dependencies not available, skipping plots")

    # -------------------------------------------------------------------------
    # 7. Compare with and without context (if both exist)
    # -------------------------------------------------------------------------
    other_suffix = "no_context" if use_context else "with_context"
    other_json = output_dir / f"shap_results_{other_suffix}.json"
    if other_json.exists():
        with open(other_json) as f:
            other = json.load(f)

        print(f"\n{'=' * 70}")
        print("COMPARISON: WITH vs WITHOUT CONTEXT")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<20} {'With Context':>15} {'No Context':>15}")
        print(f"  {'-' * 50}")

        if use_context:
            ctx_r2 = test_r2
            no_ctx_r2 = other["model_performance"]["test_r2"]
        else:
            no_ctx_r2 = test_r2
            ctx_r2 = other["model_performance"]["test_r2"]

        print(f"  {'Test R²':<20} {ctx_r2:>15.4f} {no_ctx_r2:>15.4f}")
        print(f"  {'R² improvement':<20} {ctx_r2 - no_ctx_r2:>+15.4f}")
        print("\n  If R² improves substantially with context, the model was previously")
        print("  confounding molecular signal with assay differences.")


if __name__ == "__main__":
    main()
