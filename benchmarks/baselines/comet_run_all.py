#!/usr/bin/env python3
"""Run COMET inference on all exported study data with conformer caching.

This wraps comet_infer.py's functions but precomputes mol.lmdb once per
study (across all seeds) to avoid redundant 3D conformer generation.

Usage (from LNPBO repo root, using COMET venv):

    /path/to/COMET/.venv/bin/python benchmarks/baselines/comet_run_all.py \
        --comet-repo /path/to/COMET \
        --weights-dir /path/to/COMET/experiments/weights/weights \
        --nthreads 4 \
        --resume

Then compute metrics (using LNPBO venv or any Python with numpy):

    python -m benchmarks.baselines.comet_baseline
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# Import comet_infer functions (this also patches torch.load)
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
from comet_infer import (
    build_infer_lmdb,
    build_mol_lmdb,
    run_comet_inference,
)


def main():
    parser = argparse.ArgumentParser(description="Run COMET inference on all exported data with conformer caching")
    parser.add_argument(
        "--comet-repo",
        type=str,
        required=True,
        help="Path to COMET repository root",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Path to COMET weights. If contains multiple fold subdirs, uses ensemble.",
    )
    parser.add_argument(
        "--exported-dir",
        type=str,
        default="benchmark_results/baselines/comet/exported_data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/baselines/comet/predictions",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="benchmark_results/baselines/comet/mol_cache",
        help="Directory for cached mol.lmdb files (one per study)",
    )
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--studies", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    args = parser.parse_args()

    exported_dir = Path(args.exported_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    comet_repo = args.comet_repo
    comet_experiments = os.path.join(comet_repo, "experiments")

    # Find all exported files
    exported_files = sorted(exported_dir.glob("*.json"))
    if not exported_files:
        print(f"No exported files found in {exported_dir}")
        return

    # Group by study
    study_files = defaultdict(list)
    for f in exported_files:
        stem = f.stem
        parts = stem.rsplit("_s", 1)
        if len(parts) == 2:
            study_id = parts[0]
            seed = int(parts[1])
            study_files[study_id].append((seed, f))

    # Filter
    if args.studies:
        target = set(args.studies.split(","))
        study_files = {k: v for k, v in study_files.items() if k in target}
    if args.seeds:
        target_seeds = set(int(s) for s in args.seeds.split(","))
        study_files = {k: [(s, f) for s, f in v if s in target_seeds] for k, v in study_files.items()}
        study_files = {k: v for k, v in study_files.items() if v}

    # Filter already done if resuming
    if args.resume:
        filtered = {}
        for sid, seed_files in study_files.items():
            remaining = []
            for seed, fpath in seed_files:
                pred_path = output_dir / f"{sid}_s{seed}_zero_shot.json"
                if not pred_path.exists():
                    remaining.append((seed, fpath))
            if remaining:
                filtered[sid] = remaining
        study_files = filtered

    total_files = sum(len(v) for v in study_files.values())
    print(f"Studies: {len(study_files)}, Files to process: {total_files}")
    print(f"COMET repo: {comet_repo}")
    print(f"Weights: {args.weights_dir}")

    if total_files == 0:
        print("Nothing to do.")
        return

    # Find weights (single fold or ensemble)
    weights_dir = Path(args.weights_dir)
    fold_dirs = None
    if not (weights_dir / "checkpoint_best.pt").exists():
        all_fold_dirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and (d / "checkpoint_best.pt").exists()])
        lipid_dirs = [d for d in all_fold_dirs if "fig3d" in d.name]
        if lipid_dirs:
            fold_dirs = [str(d) for d in lipid_dirs]
            print(f"Ensemble mode: {len(fold_dirs)} lipid LNP folds")
        elif all_fold_dirs:
            fold_dirs = [str(d) for d in all_fold_dirs]
            print(f"Ensemble mode: {len(fold_dirs)} folds")
        else:
            print(f"No checkpoint_best.pt found under {weights_dir}")
            return
    else:
        fold_dirs = [str(weights_dir)]
        print("Single fold mode")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_name = "in_house_lnp_master_schema_NPratio_AOvolratio.json"
    completed = 0

    for study_id, seed_files in sorted(study_files.items()):
        print(f"\n{'=' * 70}")
        print(f"Study: {study_id} ({len(seed_files)} seeds)")
        print(f"{'=' * 70}")

        # Step 1: Collect ALL unique SMILES across all seeds for this study
        all_smiles = set()
        study_data = {}
        for seed, fpath in seed_files:
            with open(fpath) as f:
                data = json.load(f)
            study_data[seed] = data
            for entry in data["formulations"]["seed"] + data["formulations"]["pool"]:
                for comp in entry["components"]:
                    all_smiles.add(comp["smi"])

        all_smiles = sorted(all_smiles)  # deterministic ordering
        print(f"Total unique SMILES across all seeds: {len(all_smiles)}")

        # Step 2: Build or load cached mol.lmdb
        mol_cache_path = cache_dir / f"{study_id}_mol.lmdb"
        smi2mol_id_cache_path = cache_dir / f"{study_id}_smi2mol_id.json"

        if mol_cache_path.exists() and smi2mol_id_cache_path.exists():
            print(f"Using cached mol.lmdb from {mol_cache_path}")
            with open(smi2mol_id_cache_path) as f:
                smi2mol_id = json.load(f)
            n_mols = len(smi2mol_id)
        else:
            print(f"Building mol.lmdb for {len(all_smiles)} molecules...")
            t0 = time.time()
            n_mols, smi2mol_id = build_mol_lmdb(all_smiles, str(mol_cache_path), nthreads=args.nthreads)
            elapsed = time.time() - t0
            print(f"Built mol.lmdb: {n_mols} molecules in {elapsed:.1f}s")

            # Save smi2mol_id mapping
            with open(smi2mol_id_cache_path, "w") as f:
                json.dump(smi2mol_id, f)

        # Step 3: For each seed, build infer.lmdb and run inference
        for seed, _fpath in seed_files:
            completed += 1
            data = study_data[seed]
            pool_entries = data["formulations"]["pool"]

            print(f"\n  [{completed}/{total_files}] {study_id}_s{seed}")
            print(f"    Pool: {len(pool_entries)} formulations")

            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                # Set up directory structure
                shutil.copy(
                    os.path.join(comet_experiments, "dict.txt"),
                    os.path.join(tmpdir, "dict.txt"),
                )
                task_schemas_dir = os.path.join(tmpdir, "task_schemas")
                os.makedirs(task_schemas_dir)
                shutil.copy(
                    os.path.join(comet_experiments, "task_schemas", schema_name),
                    os.path.join(task_schemas_dir, schema_name),
                )

                task_name = "lnpbo_study"
                subdataset_name = "in_house_lnp"
                task_dir = os.path.join(tmpdir, task_name)
                subdataset_dir = os.path.join(task_dir, subdataset_name)
                os.makedirs(subdataset_dir)

                # Copy cached mol.lmdb
                shutil.copy(str(mol_cache_path), os.path.join(task_dir, "mol.lmdb"))

                # Build infer.lmdb for this seed's pool
                infer_lmdb_path = os.path.join(subdataset_dir, "infer.lmdb")
                build_infer_lmdb(pool_entries, smi2mol_id, infer_lmdb_path)

                # Run inference with each fold
                all_scores = {}  # lnp_id -> list of scores
                for fold_i, wd in enumerate(fold_dirs):
                    weights_path = os.path.join(wd, "checkpoint_best.pt")
                    if not os.path.exists(weights_path):
                        continue

                    t0 = time.time()
                    predictions, _ = run_comet_inference(
                        comet_repo=comet_repo,
                        weights_path=weights_path,
                        task_schema_path=os.path.join("task_schemas", schema_name),
                        data_dir=tmpdir,
                        task_name=task_name,
                    )
                    elapsed = time.time() - t0

                    n_preds = len(predictions)
                    if n_preds > 0:
                        score_range = f"[{min(predictions.values()):.3f}, {max(predictions.values()):.3f}]"
                    else:
                        score_range = "N/A"
                    print(f"    Fold {fold_i}: {n_preds} preds, range={score_range}, {elapsed:.1f}s")

                    for lnp_id, score in predictions.items():
                        if lnp_id not in all_scores:
                            all_scores[lnp_id] = []
                        all_scores[lnp_id].append(score)

                # Average across folds
                pred_list = []
                for entry in pool_entries:
                    lnp_id = str(entry["idx"])
                    scores = all_scores.get(lnp_id, [0.0])
                    avg_score = sum(scores) / len(scores)
                    pred_list.append({"idx": entry["idx"], "score": avg_score})

                # Save predictions
                out_path = output_dir / f"{study_id}_s{seed}_zero_shot.json"
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "predictions": pred_list,
                            "n_folds": len(fold_dirs),
                        },
                        f,
                        indent=2,
                    )
                print(f"    Saved to {out_path}")

    print(f"\n{'=' * 70}")
    print(f"Done. Processed {completed} files.")
    print("Run 'python -m benchmarks.baselines.comet_baseline' to compute metrics.")


if __name__ == "__main__":
    main()
