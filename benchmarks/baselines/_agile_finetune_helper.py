#!/usr/bin/env python3
"""AGILE fine-tuning + inference helper.

Standalone script invoked via subprocess from agile_baseline.py.
Must be run with an environment that has torch, torch_geometric,
torch_scatter, and rdkit installed (i.e. the AGILE repo venv).

Usage:
    python _agile_finetune_helper.py \\
        --train-csv /tmp/train.csv \\
        --pool-csv /tmp/pool.csv \\
        --output-json /tmp/predictions.json \\
        --agile-root /path/to/AGILE \\
        --epochs 30 \\
        --seed 42

The train CSV must have columns: smiles, label
The pool CSV must have columns: smiles, label (label can be dummy 0)

Output JSON contains:
    {"pool_predictions": [float, ...], "train_predictions": [float, ...]}
"""

import argparse
import copy
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--pool-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--agile-root", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--base-lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-ratio", type=float, default=0.3)
    args = parser.parse_args()

    # Add AGILE repo to path
    sys.path.insert(0, args.agile_root)

    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from dataset.dataset_test import MolTestDataset
    from models.agile_finetune import AGILE
    from torch_geometric.data import DataLoader

    device = torch.device("cpu")

    # Load training data
    print(f"Loading training data from {args.train_csv}")
    train_dataset = MolTestDataset(args.train_csv, target="label", task="regression")
    print(f"  {len(train_dataset)} training molecules")

    # Load pool data
    print(f"Loading pool data from {args.pool_csv}")
    pool_dataset = MolTestDataset(args.pool_csv, target="label", task="regression")
    print(f"  {len(pool_dataset)} pool molecules")

    if len(train_dataset) == 0:
        print("ERROR: Empty training set")
        result = {
            "pool_predictions": [0.0] * len(pool_dataset),
            "train_predictions": [],
        }
        with open(args.output_json, "w") as f:
            json.dump(result, f)
        return

    # Split training into train/val (90/10)
    n_train = len(train_dataset)
    indices = list(range(n_train))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(0.1 * n_train))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    from torch.utils.data.sampler import SubsetRandomSampler

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,
        drop_last=False,
    )

    # Load pretrained AGILE model
    ckpt_path = os.path.join(args.agile_root, "ckpt", "pretrained_agile_60k", "checkpoints", "model.pth")
    model = AGILE(
        task="regression",
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=args.drop_ratio,
        pool="mean",
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_my_state_dict(state_dict)
    model.to(device)
    print("Loaded pretrained AGILE weights")

    # Set up optimizer with different LR for prediction head vs base
    layer_list = []
    for name, _param in model.named_parameters():
        if "pred_" in name:
            layer_list.append(name)

    params = list(
        map(
            lambda x: x[1],
            list(filter(lambda kv: kv[0] in layer_list, model.named_parameters())),
        )
    )
    base_params = list(
        map(
            lambda x: x[1],
            list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters())),
        )
    )

    optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": args.base_lr},
            {"params": params},
        ],
        args.lr,
        weight_decay=1e-6,
    )

    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_model_state = None
    patience = 5
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_samples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            _, pred = model(data)
            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.y.size(0)
            n_samples += data.y.size(0)

        train_loss /= max(n_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                _, pred = model(data)
                loss = criterion(pred, data.y)
                val_loss += loss.item() * data.y.size(0)
                n_val_samples += data.y.size(0)

        val_loss /= max(n_val_samples, 1)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"  Best validation loss: {best_val_loss:.4f}")

    # Inference on pool
    pool_loader = DataLoader(
        pool_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model.eval()
    pool_preds = []
    with torch.no_grad():
        for data in pool_loader:
            data = data.to(device)
            _, pred = model(data)
            pool_preds.extend(pred.cpu().numpy().flatten().tolist())

    # Also predict on training data for diagnostics
    full_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    train_preds = []
    with torch.no_grad():
        for data in full_train_loader:
            data = data.to(device)
            _, pred = model(data)
            train_preds.extend(pred.cpu().numpy().flatten().tolist())

    result = {
        "pool_predictions": pool_preds,
        "train_predictions": train_preds,
        "best_val_loss": float(best_val_loss),
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f)
    print(f"Saved predictions to {args.output_json}")


if __name__ == "__main__":
    main()
