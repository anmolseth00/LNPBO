#!/usr/bin/env python3
"""Train GPS-MPNN model on LNPDB data.

GPS-style hybrid: D-MPNN + RWSE positional encoding + global self-attention
+ attention readout + cross-component attention.

Usage:
    python models/train_gps.py
    python models/train_gps.py --components IL HL CHL PEG --depth 4
"""


import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from featurize import ATOM_FDIM, BOND_FDIM, BatchMolGraph
from gps_mpnn import MultiComponentGPS, make_graph_fn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from data import (
    TABULAR_CONTINUOUS_COLS,
    LNPDataset,
    encode_categoricals,
    learn_categorical_levels,
    load_lnpdb_dataframe,
    make_dataloader,
    scaffold_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPS-MPNN on LNPDB", suggest_on_error=True)

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--components", nargs="+", default=["IL"],
                        choices=["IL", "HL", "CHL", "PEG"])
    parser.add_argument("--split", type=str, default="scaffold",
                        choices=["scaffold", "random"])
    parser.add_argument("--split-seed", type=int, default=42)

    # Model
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-attn-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=256)
    parser.add_argument("--ffn-num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rwse-dim", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-frac", type=float, default=0.06)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--save-dir", type=str, default="models/runs/gps_baseline")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-categorical", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    component_graphs: dict[str, BatchMolGraph],
    tabular: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, BatchMolGraph], torch.Tensor, torch.Tensor]:
    moved = {}
    for name, bg in component_graphs.items():
        moved[name] = BatchMolGraph(
            f_atoms=bg.f_atoms.to(device),
            f_bonds=bg.f_bonds.to(device),
            a2b=bg.a2b.to(device),
            b2a=bg.b2a.to(device),
            b2revb=bg.b2revb.to(device),
            a_scope=bg.a_scope,
            n_mols=bg.n_mols,
        )
    return moved, tabular.to(device), targets.to(device)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for component_graphs, tabular, targets in loader:
        component_graphs, tabular, targets = move_batch_to_device(
            component_graphs, tabular, targets, device
        )
        optimizer.zero_grad()
        preds = model(component_graphs, tabular).squeeze(-1)
        loss = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_mean=0.0, target_std=1.0):
    model.eval()
    all_preds = []
    all_targets = []

    for component_graphs, tabular, targets in loader:
        component_graphs, tabular, targets = move_batch_to_device(
            component_graphs, tabular, targets, device
        )
        preds = model(component_graphs, tabular).squeeze(-1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    preds_orig = preds * target_std + target_mean
    targets_orig = targets * target_std + target_mean

    mse = float(np.mean((preds_orig - targets_orig) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds_orig - targets_orig)))

    ss_res = np.sum((targets_orig - preds_orig) ** 2)
    ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "preds": preds_orig, "targets": targets_orig}


@torch.no_grad()
def extract_fingerprints(model, loader, device):
    model.eval()
    fps = []
    for component_graphs, tabular, _ in loader:
        component_graphs, tabular, _ = move_batch_to_device(
            component_graphs, tabular, torch.zeros(tabular.size(0)), device
        )
        fp = model.encode(component_graphs, tabular)
        fps.append(fp.cpu().numpy())
    return np.concatenate(fps, axis=0)


def save_plots(train_losses, val_rmses, test_results, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss", alpha=0.8)
    ax.plot(epochs_range, val_rmses, label="Val RMSE", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / RMSE")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    preds = test_results["preds"]
    targets = test_results["targets"]
    ax.scatter(targets, preds, alpha=0.3, s=8, edgecolors="none")
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Test: RMSE={test_results['rmse']:.3f}, R2={test_results['r2']:.3f}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    residuals = preds - targets
    ax.scatter(targets, residuals, alpha=0.3, s=8, edgecolors="none")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals (MAE={test_results['mae']:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {save_dir / 'training_results.png'}")


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    df = load_lnpdb_dataframe(args.data_path, components=args.components)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    print(f"After filtering: {len(df)} rows")

    tabular_cols = [c for c in TABULAR_CONTINUOUS_COLS if c in df.columns]

    # Split
    print(f"Splitting ({args.split})...")
    if args.split == "scaffold":
        il_smiles = df["IL_SMILES"].tolist()
        train_idx, val_idx, test_idx = scaffold_split(
            il_smiles, sizes=(0.8, 0.1, 0.1), seed=args.split_seed
        )
    else:
        n = len(df)
        rng = np.random.RandomState(args.split_seed)
        indices = rng.permutation(n).tolist()
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Categoricals
    categorical_cols: list[str] = []
    cat_levels: dict[str, list[str]] = {}
    if not args.no_categorical:
        cat_levels = learn_categorical_levels(df.iloc[train_idx])
        df, categorical_cols = encode_categoricals(df, cat_levels)
        print(f"Categorical features ({len(categorical_cols)} one-hot)")

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]

    # Normalization
    target_mean = float(df_train["Experiment_value"].mean())
    target_std = float(df_train["Experiment_value"].std())
    if target_std < 1e-8:
        target_std = 1.0
    print(f"Target stats: mean={target_mean:.4f}, std={target_std:.4f}")

    tabular_mean = None
    tabular_std = None
    if tabular_cols:
        tab_train = df_train[tabular_cols].fillna(0.0).values.astype(np.float32)
        tabular_mean = tab_train.mean(axis=0)
        tabular_std = tab_train.std(axis=0)

    # Build datasets with RWSE-enhanced graph featurization
    graph_fn = make_graph_fn(rwse_dim=args.rwse_dim)
    atom_fdim = ATOM_FDIM + args.rwse_dim

    print(f"Building datasets (featurizing molecules with RWSE dim={args.rwse_dim})...")
    t0 = time.time()

    ds_train = LNPDataset(
        df_train, components=args.components, tabular_cols=tabular_cols,
        categorical_cols=categorical_cols,
        tabular_mean=tabular_mean, tabular_std=tabular_std,
        target_mean=target_mean, target_std=target_std,
        graph_fn=graph_fn,
    )
    ds_val = LNPDataset(
        df_val, components=args.components, tabular_cols=tabular_cols,
        categorical_cols=categorical_cols,
        tabular_mean=tabular_mean, tabular_std=tabular_std,
        target_mean=target_mean, target_std=target_std,
        graph_fn=graph_fn,
    )
    ds_test = LNPDataset(
        df_test, components=args.components, tabular_cols=tabular_cols,
        categorical_cols=categorical_cols,
        tabular_mean=tabular_mean, tabular_std=tabular_std,
        target_mean=target_mean, target_std=target_std,
        graph_fn=graph_fn,
    )

    print(f"Featurization done in {time.time() - t0:.1f}s")

    loader_train = make_dataloader(ds_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    loader_val = make_dataloader(ds_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    loader_test = make_dataloader(ds_test, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    # Model
    tabular_dim = len(tabular_cols) + len(categorical_cols)

    model = MultiComponentGPS(
        component_names=args.components,
        atom_fdim=atom_fdim,
        bond_fdim=BOND_FDIM,
        hidden_size=args.hidden_size,
        depth=args.depth,
        n_attn_heads=args.n_attn_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        ffn_num_layers=args.ffn_num_layers,
        tabular_dim=tabular_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")
    print(f"Components: {args.components}, hidden={args.hidden_size}, depth={args.depth}, "
          f"heads={args.n_attn_heads}, rwse_dim={args.rwse_dim}, tabular_dim={tabular_dim}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader_train)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_frac, anneal_strategy="cos",
        final_div_factor=args.lr / 1e-5,
    )
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"{'Epoch':>5} | {'Train Loss':>11} | {'Val RMSE':>9} | {'Val MAE':>8} | "
          f"{'Val R2':>7} | {'LR':>10} | {'Time':>6}")
    print("-" * 75)

    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_rmses = []

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_loss = train_epoch(
            model, loader_train, optimizer, scheduler, criterion, device, args.grad_clip
        )
        val_results = evaluate(
            model, loader_val, criterion, device,
            target_mean=target_mean, target_std=target_std,
        )

        t_elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_rmses.append(val_results["rmse"])

        print(f"{epoch:5d} | {train_loss:11.4f} | {val_results['rmse']:9.4f} | "
              f"{val_results['mae']:8.4f} | {val_results['r2']:7.4f} | "
              f"{current_lr:10.2e} | {t_elapsed:5.1f}s")

        if val_results["rmse"] < best_val_rmse:
            best_val_rmse = val_results["rmse"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "args": vars(args),
                "target_mean": target_mean,
                "target_std": target_std,
                "tabular_mean": tabular_mean.tolist() if tabular_mean is not None else None,
                "tabular_std": tabular_std.tolist() if tabular_std is not None else None,
                "tabular_cols": tabular_cols,
                "categorical_cols": categorical_cols,
                "cat_levels": cat_levels,
                "components": args.components,
                "atom_fdim": atom_fdim,
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best={best_epoch}, "
                      f"val_rmse={best_val_rmse:.4f})")
                break

    # Evaluate best model on test
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results = evaluate(
        model, loader_test, criterion, device,
        target_mean=target_mean, target_std=target_std,
    )

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (best epoch {best_epoch}):")
    print(f"  RMSE:  {test_results['rmse']:.4f}")
    print(f"  MAE:   {test_results['mae']:.4f}")
    print(f"  R2:    {test_results['r2']:.4f}")
    print(f"{'='*60}")

    metrics = {
        "test_rmse": float(test_results["rmse"]),
        "test_mae": float(test_results["mae"]),
        "test_r2": float(test_results["r2"]),
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_val_rmse),
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_params": n_params,
        "args": vars(args),
    }
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Fingerprints
    print("\nExtracting fingerprints from test set...")
    fps = extract_fingerprints(model, loader_test, device)
    print(f"Fingerprint shape: {fps.shape}")
    np.save(save_dir / "test_fingerprints.npy", fps)

    if not args.no_plots:
        try:
            save_plots(train_losses, val_rmses, test_results, save_dir)
        except ImportError:
            print("matplotlib not available, skipping plots")

    print(f"\nAll outputs saved to {save_dir}/")


if __name__ == "__main__":
    main()
