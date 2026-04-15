#!/usr/bin/env python3
"""AGILE fine-tuned predictor baseline.

Two modes:
  1. "mlp" (default): Fine-tunes an MLP prediction head on top of frozen
     pretrained AGILE GNN embeddings. Uses pre-computed 512-d embeddings
     from data/agile_embeddings.npz. Runs in the LNPBO venv (no
     torch_geometric needed).

  2. "gnn": End-to-end fine-tunes the full AGILE GNN on each study's seed
     data. Requires torch_geometric and is invoked via subprocess using
     the AGILE repo's Python environment.

Reference: Xu et al., "AGILE: A Graph Neural Network Framework for
Ionizable Lipid Design", Nature Communications, 2024.

Usage:
    python -m benchmarks.baselines.agile_baseline
    python -m benchmarks.baselines.agile_baseline --mode mlp
    python -m benchmarks.baselines.agile_baseline --mode gnn
    python -m benchmarks.baselines.agile_baseline --resume
    python -m benchmarks.baselines.agile_baseline --aggregate-only
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from LNPBO.benchmarks.benchmark import (
    BATCH_SIZE,
    MAX_ROUNDS,
    SEEDS,
    ensure_top_k_pct,
    filter_study_df,
    get_study_id,
)
from LNPBO.benchmarks.runner import compute_metrics, init_history, update_history
from LNPBO.benchmarks.stats import bootstrap_ci
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)
RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "baselines" / "agile_finetuned"

AGILE_ROOT = Path(os.environ.get("AGILE_ROOT", Path.home() / "Documents" / "GitHub" / "AGILE"))
AGILE_PYTHON = os.environ.get("AGILE_PYTHON", str(AGILE_ROOT / ".venv" / "bin" / "python"))
HELPER_SCRIPT = Path(__file__).resolve().parent / "_agile_finetune_helper.py"

FINETUNE_EPOCHS = 50
MLP_HIDDEN = 256


def _per_seed_path(study_id, seed, mode="mlp"):
    return RESULTS_DIR / study_id / f"agile_finetuned_{mode}_s{seed}.json"


def _prepare_study_split(df, study_info, random_seed):
    """Prepare seed/oracle split for a study."""
    study_df = filter_study_df(df, study_info)
    n_seed = study_info["n_seed"]

    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(study_df))
    rng.shuffle(all_idx)

    seed_idx = sorted(all_idx[:n_seed])
    oracle_idx = sorted(all_idx[n_seed:])

    top_k_values = {}
    for pct, k in study_info["top_k_pct"].items():
        actual_k = min(k, len(study_df))
        top_k_values[pct] = set(study_df.nlargest(actual_k, "Experiment_value").index)

    return study_df, seed_idx, oracle_idx, top_k_values


def _load_agile_embedding_cache():
    """Load AGILE embeddings and return SMILES->embedding dict."""
    npz_path = Path(__file__).resolve().parent.parent.parent / "data" / "agile_embeddings.npz"
    data = np.load(npz_path, allow_pickle=False)
    smiles = data["smiles"]
    embeddings = data["embeddings"]
    cache = {str(s): embeddings[i] for i, s in enumerate(smiles)}
    print(f"  Loaded {len(cache)} AGILE embeddings ({embeddings.shape[1]}-d)")
    return cache, embeddings.shape[1]


def _get_embeddings(cache, embed_dim, smiles_list):
    """Look up embeddings for SMILES, returning (embeddings, valid_mask)."""
    embs = []
    valid = []
    for s in smiles_list:
        if pd.notna(s) and s in cache:
            embs.append(cache[s])
            valid.append(True)
        else:
            embs.append(np.zeros(embed_dim))
            valid.append(False)
    return np.array(embs), np.array(valid)


def run_agile_mlp_finetuned(
    study_df,
    seed_idx,
    oracle_idx,
    top_k_values,
    batch_size,
    n_rounds,
    random_seed,
    agile_cache,
    embed_dim,
    epochs=50,
):
    """Fine-tune MLP on frozen AGILE embeddings, predict on oracle pool."""
    import torch
    from torch import nn

    torch.manual_seed(random_seed)

    # Get embeddings for seed and oracle
    seed_smiles = study_df.iloc[seed_idx]["IL_SMILES"].values
    oracle_smiles = study_df.iloc[oracle_idx]["IL_SMILES"].values

    X_seed, seed_valid = _get_embeddings(agile_cache, embed_dim, seed_smiles)
    X_oracle, oracle_valid = _get_embeddings(agile_cache, embed_dim, oracle_smiles)

    y_seed = study_df.iloc[seed_idx]["Experiment_value"].values

    # Filter to valid embeddings only
    valid_seed_mask = seed_valid
    valid_oracle_mask = oracle_valid

    X_train_all = X_seed[valid_seed_mask].astype(np.float32)
    y_train_all = y_seed[valid_seed_mask].astype(np.float32)
    X_pool = X_oracle[valid_oracle_mask].astype(np.float32)

    valid_oracle_indices = [oracle_idx[i] for i in range(len(oracle_idx)) if valid_oracle_mask[i]]

    if len(X_train_all) < 10 or len(X_pool) == 0:
        raise ValueError(f"Too few valid embeddings: train={len(X_train_all)}, pool={len(X_pool)}")

    # Concat ratio features if available
    ratio_cols = [
        c
        for c in ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]
        if c in study_df.columns and study_df[c].nunique() > 1
    ]
    if ratio_cols:
        seed_ratios = study_df.iloc[seed_idx][ratio_cols].values[valid_seed_mask].astype(np.float32)
        oracle_ratios = study_df.iloc[oracle_idx][ratio_cols].values[valid_oracle_mask].astype(np.float32)
        # Normalize ratios
        r_mean = seed_ratios.mean(axis=0)
        r_std = seed_ratios.std(axis=0)
        r_std[r_std == 0] = 1.0
        seed_ratios = (seed_ratios - r_mean) / r_std
        oracle_ratios = (oracle_ratios - r_mean) / r_std
        X_train_all = np.hstack([X_train_all, seed_ratios])
        X_pool = np.hstack([X_pool, oracle_ratios])

    input_dim = X_train_all.shape[1]

    # Normalize embeddings
    emb_mean = X_train_all.mean(axis=0)
    emb_std = X_train_all.std(axis=0)
    emb_std[emb_std == 0] = 1.0
    X_train_all = (X_train_all - emb_mean) / emb_std
    X_pool = (X_pool - emb_mean) / emb_std

    # Normalize targets
    y_mean = y_train_all.mean()
    y_std = y_train_all.std()
    if y_std == 0:
        y_std = 1.0
    y_train_norm = (y_train_all - y_mean) / y_std

    # Train/val split
    rng = np.random.RandomState(random_seed)
    n = len(X_train_all)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_perm = perm[:n_val]
    train_perm = perm[n_val:]

    X_train = torch.tensor(X_train_all[train_perm])
    y_train = torch.tensor(y_train_norm[train_perm]).unsqueeze(1)
    X_val = torch.tensor(X_train_all[val_perm])
    y_val = torch.tensor(y_train_norm[val_perm]).unsqueeze(1)

    # Build MLP (matches AGILE's pred_head architecture)
    model = nn.Sequential(
        nn.Linear(input_dim, MLP_HIDDEN),
        nn.Softplus(),
        nn.Dropout(0.3),
        nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
        nn.Softplus(),
        nn.Dropout(0.3),
        nn.Linear(MLP_HIDDEN, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on pool
    model.eval()
    X_pool_t = torch.tensor(X_pool)
    with torch.no_grad():
        pool_scores = model(X_pool_t).numpy().flatten()

    # Rank and select
    total_budget = n_rounds * batch_size
    ranked_positions = np.argsort(-pool_scores)
    n_select = min(total_budget, len(valid_oracle_indices))
    selected_positions = ranked_positions[:n_select]

    training_idx = list(seed_idx)
    history = init_history(study_df, training_idx, top_k_values=top_k_values)

    for r in range(n_rounds):
        start = r * batch_size
        end = min((r + 1) * batch_size, n_select)
        if start >= n_select:
            break
        batch_positions = selected_positions[start:end]
        batch_idx = [valid_oracle_indices[i] for i in batch_positions]
        training_idx.extend(batch_idx)
        update_history(history, study_df, training_idx, batch_idx, r, top_k_values=top_k_values)

    return history, best_val_loss


def _write_csv(study_df, indices, path):
    """Write a CSV suitable for AGILE's MolTestDataset."""
    sub = study_df.iloc[indices]
    out = pd.DataFrame({"smiles": sub["IL_SMILES"].values, "label": sub["Experiment_value"].values})
    out = out.dropna(subset=["smiles"])
    out = out[out["smiles"].str.len() > 0]
    out.to_csv(path, index=False)
    return len(out)


def run_agile_gnn_finetuned(
    study_df,
    seed_idx,
    oracle_idx,
    top_k_values,
    batch_size,
    n_rounds,
    random_seed,
    epochs=30,
):
    """Fine-tune full AGILE GNN via subprocess, predict on oracle pool."""

    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        pool_csv = os.path.join(tmpdir, "pool.csv")
        output_json = os.path.join(tmpdir, "predictions.json")

        n_train = _write_csv(study_df, seed_idx, train_csv)
        n_pool = _write_csv(study_df, oracle_idx, pool_csv)

        if n_train == 0 or n_pool == 0:
            raise ValueError(f"Empty split: n_train={n_train}, n_pool={n_pool}")

        cmd = [
            AGILE_PYTHON,
            str(HELPER_SCRIPT),
            "--train-csv",
            train_csv,
            "--pool-csv",
            pool_csv,
            "--output-json",
            output_json,
            "--agile-root",
            str(AGILE_ROOT),
            "--epochs",
            str(epochs),
            "--batch-size",
            "64",
            "--seed",
            str(random_seed),
        ]

        print(f"    Running AGILE GNN fine-tuning (n_train={n_train}, n_pool={n_pool})...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(AGILE_ROOT))

        if proc.returncode != 0:
            print(f"    STDERR:\n{proc.stderr[-500:]}")
            raise RuntimeError(f"AGILE fine-tuning failed (rc={proc.returncode})")

        for line in proc.stdout.strip().split("\n")[-5:]:
            print(f"    {line}")

        with open(output_json) as f:
            predictions = json.load(f)

    pool_scores = np.array(predictions["pool_predictions"])
    best_val_loss = predictions.get("best_val_loss", float("nan"))

    pool_smiles = study_df.iloc[oracle_idx]["IL_SMILES"].values
    valid_pool_mask = pd.notna(pool_smiles) & (pd.Series(pool_smiles).str.len() > 0).values
    valid_oracle_idx = [oracle_idx[i] for i in range(len(oracle_idx)) if valid_pool_mask[i]]

    if len(pool_scores) != len(valid_oracle_idx):
        n_preds = len(pool_scores)
        print(f"    Warning: {n_preds} predictions for {len(valid_oracle_idx)} valid pool molecules")
        valid_oracle_idx = valid_oracle_idx[:n_preds]

    total_budget = n_rounds * batch_size
    ranked_positions = np.argsort(-pool_scores)
    n_select = min(total_budget, len(valid_oracle_idx))
    selected_positions = ranked_positions[:n_select]

    training_idx = list(seed_idx)
    history = init_history(study_df, training_idx, top_k_values=top_k_values)

    for r in range(n_rounds):
        start = r * batch_size
        end = min((r + 1) * batch_size, n_select)
        if start >= n_select:
            break
        batch_positions = selected_positions[start:end]
        batch_idx = [valid_oracle_idx[i] for i in batch_positions]
        training_idx.extend(batch_idx)
        update_history(history, study_df, training_idx, batch_idx, r, top_k_values=top_k_values)

    return history, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="AGILE Fine-Tuned Predictor Baseline")
    parser.add_argument(
        "--mode",
        type=str,
        default="mlp",
        choices=["mlp", "gnn"],
        help="mlp = MLP on frozen AGILE embeddings; gnn = full GNN fine-tuning",
    )
    parser.add_argument("--pmids", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--studies-json", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    mode = args.mode
    epochs = args.epochs or (FINETUNE_EPOCHS if mode == "mlp" else 30)

    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]

    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations")

    # Load study definitions
    if args.studies_json:
        with open(args.studies_json) as f:
            study_infos = json.load(f)
    else:
        default_studies = (
            Path(__file__).resolve().parent.parent.parent / "experiments" / "data_integrity" / "studies.json"
        )
        if default_studies.exists():
            with open(default_studies) as f:
                study_infos = json.load(f)
        else:
            from LNPBO.benchmarks.benchmark import characterize_studies

            study_infos = characterize_studies(df)

    # Filter out ratio-only studies (AGILE requires molecular features)
    study_infos = [s for s in study_infos if s.get("study_type") != "ratio_only"]

    if args.pmids:
        target = set(args.pmids.split(","))
        study_infos = [
            si for si in study_infos if si.get("study_id", str(si["pmid"])) in target or str(si["pmid"]) in target
        ]

    ensure_top_k_pct(study_infos)

    # Determine runs needed
    runs = []
    for si in study_infos:
        for seed in seeds:
            sid = get_study_id(si)
            path = _per_seed_path(sid, seed, mode)
            if (args.resume or args.aggregate_only) and path.exists():
                continue
            if args.aggregate_only:
                continue
            runs.append((si, seed))

    print(f"\n{'=' * 70}")
    print(f"AGILE FINE-TUNED BASELINE (mode={mode})")
    print(f"{'=' * 70}")
    print(f"Studies: {len(study_infos)}")
    print(f"Seeds: {seeds}")
    print(f"Runs needed: {len(runs)}")
    print(f"Mode: {mode}")
    print(f"Epochs: {epochs}")

    if args.dry_run:
        for si, seed in runs[:20]:
            sid = get_study_id(si)
            print(f"  {sid} / seed={seed} (N={si['n_formulations']})")
        if len(runs) > 20:
            print(f"  ... and {len(runs) - 20} more")
        return

    # Pre-load AGILE embeddings for MLP mode
    agile_cache = None
    embed_dim = None
    if mode == "mlp":
        agile_cache, embed_dim = _load_agile_embedding_cache()

    # Run
    if runs:
        print(f"\nRunning {len(runs)} study-seed combinations...\n")

        groups = defaultdict(list)
        for si, seed in runs:
            groups[get_study_id(si)].append((si, seed))

        completed = 0
        total = len(runs)

        for sid, group in groups.items():
            si = group[0][0]
            print(f"\n--- Study {sid} (N={si['n_formulations']}, ILs={si['n_unique_il']}, type={si['study_type']}) ---")

            for si, seed in group:
                completed += 1
                print(f"\n  [{completed}/{total}] seed={seed}")

                try:
                    t0 = time.time()
                    study_df, seed_idx, oracle_idx, topk = _prepare_study_split(df, si, seed)

                    if mode == "mlp":
                        history, val_loss = run_agile_mlp_finetuned(
                            study_df,
                            seed_idx,
                            oracle_idx,
                            topk,
                            batch_size=si.get("batch_size", BATCH_SIZE),
                            n_rounds=si.get("n_rounds", MAX_ROUNDS),
                            random_seed=seed,
                            agile_cache=agile_cache,
                            embed_dim=embed_dim,
                            epochs=epochs,
                        )
                    else:
                        history, val_loss = run_agile_gnn_finetuned(
                            study_df,
                            seed_idx,
                            oracle_idx,
                            topk,
                            batch_size=si.get("batch_size", BATCH_SIZE),
                            n_rounds=si.get("n_rounds", MAX_ROUNDS),
                            random_seed=seed,
                            epochs=epochs,
                        )

                    elapsed = time.time() - t0
                    metrics = compute_metrics(history, topk, len(study_df))
                    metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

                    recall_str = ", ".join(
                        f"Top-{k}%={metrics['top_k_recall'].get(str(k), 0):.1%}" for k in [5, 10, 20]
                    )
                    print(f"    {elapsed:.1f}s | val_loss={val_loss:.4f} | {recall_str}")

                    result = {
                        "metrics": metrics,
                        "elapsed": elapsed,
                        "best_so_far": history["best_so_far"],
                        "round_best": history["round_best"],
                        "n_evaluated": history["n_evaluated"],
                        "val_loss": val_loss,
                    }

                    path = _per_seed_path(sid, seed, mode)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    out = {
                        "baseline": "agile_finetuned",
                        "surrogate": f"agile_finetuned_{mode}",
                        "study_id": sid,
                        "pmid": si["pmid"],
                        "seed": seed,
                        "study_info": {k: v for k, v in si.items() if k not in ("lnp_ids", "top_k_pct")},
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                        "mode": mode,
                        "epochs": epochs,
                    }
                    with open(path, "w") as f:
                        json.dump(out, f, indent=2, default=str)

                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback

                    traceback.print_exc()

    # Aggregation
    print(f"\n{'=' * 70}")
    print("AGGREGATION")
    print(f"{'=' * 70}")

    if not RESULTS_DIR.exists():
        print("No results to aggregate.")
        return

    all_study_means = []
    study_results = {}

    for si in study_infos:
        sid = get_study_id(si)
        vals = []
        for seed in seeds:
            path = _per_seed_path(sid, seed, mode)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                recall = data["result"]["metrics"]["top_k_recall"]
                vals.append(recall.get("5", 0.0))

        if vals:
            m = float(np.mean(vals))
            ci = bootstrap_ci(vals) if len(vals) >= 3 else (m, m)
            study_results[sid] = {
                "mean_top5": m,
                "std_top5": (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0),
                "ci": ci,
                "n_seeds": len(vals),
                "per_seed": vals,
            }
            all_study_means.append(m)

    if study_results:
        print(f"\n{'Study':<25} {'Mean Top-5%':>12} {'Std':>8} {'95% CI':>18} {'N seeds':>8}")
        print("-" * 75)
        for sid, r in sorted(study_results.items()):
            print(
                f"{sid:<25} {r['mean_top5']:>11.1%} "
                f"{r['std_top5']:>7.1%} "
                f"[{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] "
                f"{r['n_seeds']:>8}"
            )

        grand_mean = float(np.mean(all_study_means))
        grand_ci = bootstrap_ci(all_study_means) if len(all_study_means) >= 3 else (grand_mean, grand_mean)
        print("-" * 75)
        print(
            f"{'OVERALL':<25} {grand_mean:>11.1%} "
            f"{'':>7} "
            f"[{grand_ci[0]:.1%}, {grand_ci[1]:.1%}] "
            f"{len(study_results):>8}"
        )

        print("\n--- Reference Baselines ---")
        print("  Random:              0.532 (53.2%)")
        print("  AGILE Emb + XGB P&R: ~0.66")
        print("  LANTERN + XGB P&R:   ~0.68")
        print("  NGBoost BO:          0.737 (73.7%)")

        summary = {
            "baseline": "agile_finetuned",
            "mode": mode,
            "grand_mean_top5": grand_mean,
            "grand_ci": grand_ci,
            "n_studies": len(study_results),
            "per_study": study_results,
            "timestamp": datetime.now().isoformat(),
        }
        summary_path = RESULTS_DIR / f"agile_finetuned_{mode}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
