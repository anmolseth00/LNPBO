#!/usr/bin/env python3
"""COMET Baseline Wrapper.

Evaluates COMET (Composite Material Transformer) as a predict-and-rank
baseline for LNP formulation optimization. COMET uses a multi-component
Transformer architecture with Uni-Mol molecular encoders.

Three execution modes:

  Mode A (recommended): Direct local inference (works on CPU, no CUDA needed)
    python -m benchmarks.baselines.comet_wrapper --export
    python -m benchmarks.baselines.comet_wrapper --run \
        --comet-repo ~/COMET --weights-dir ~/COMET/experiments/weights/<dir>
    python -m benchmarks.baselines.comet_wrapper --import-results

  Mode B: Two-machine workflow (export on any, infer on Linux+CUDA)
    python -m benchmarks.baselines.comet_wrapper --export
    # Transfer exported_data/ to Linux machine, run comet_infer.py there
    python -m benchmarks.baselines.comet_wrapper --import-results

COMET's Uni-Core CUDA kernels are optional. The code has CPU fallbacks for
softmax_dropout, fused_adam, and fused_multi_tensor. Inference runs fine on
CPU (macOS ARM64 included) -- it is just slower.

Reference: Chang et al., "COMET: Composite Material Transformer for
Lipid Nanoparticle Design", NeurIPS 2024 ML4H Workshop.

Requirements for inference:
  - Python 3.10
  - PyTorch (CPU ok)
  - rdkit, lmdb, numpy<2, pyprojroot, pandas, scikit-learn
  - COMET repo: https://github.com/alvinchangw/COMET
  - Pretrained weights: https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from LNPBO.benchmarks.benchmark import (
    BATCH_SIZE,
    MAX_ROUNDS,
    SEEDS,
    characterize_studies,
    ensure_top_k_pct,
    get_study_id,
)
from LNPBO.benchmarks.runner import compute_metrics, init_history, update_history
from LNPBO.benchmarks.stats import bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark_results" / "baselines" / "comet"
EXPORT_DIR = RESULTS_DIR / "exported_data"


def _per_seed_path(study_id, mode, seed):
    return RESULTS_DIR / study_id / f"comet_{mode}_s{seed}.json"


def export_study_data(df, study_infos, seeds):
    """Export study data in COMET-compatible JSON format.

    For each study x seed, exports:
      - Seed formulations (training set) with component SMILES, types, ratios, and labels
      - Pool formulations (inference set) with component SMILES, types, ratios
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    smiles_cols = {
        "IL": "IL_SMILES",
        "HL": "HL_SMILES",
        "CH": "CHL_SMILES",
        "PEG": "PEG_SMILES",
    }
    ratio_cols = {
        "IL": "IL_molratio",
        "HL": "HL_molratio",
        "CH": "CHL_molratio",
        "PEG": "PEG_molratio",
    }

    exported = 0
    for si in study_infos:
        sid = get_study_id(si)
        study_df = df[df["Publication_PMID"] == float(si["pmid"])].copy()
        if len(study_df) == 0:
            continue

        for seed in seeds:
            rng = np.random.RandomState(seed)
            n_seed = si["n_seed"]
            all_idx = list(study_df.index)
            rng.shuffle(all_idx)
            seed_idx = all_idx[:n_seed]
            pool_idx = all_idx[n_seed:]

            formulations = {}
            for label, idx_list in [("seed", seed_idx), ("pool", pool_idx)]:
                entries = []
                for idx in idx_list:
                    row = study_df.loc[idx]
                    components = []
                    for comp_type in ["IL", "HL", "CH", "PEG"]:
                        smi_col = smiles_cols[comp_type]
                        rat_col = ratio_cols[comp_type]
                        if smi_col in row.index and pd.notna(row[smi_col]):
                            components.append(
                                {
                                    "smi": str(row[smi_col]),
                                    "component_type": comp_type,
                                    "mol": float(row[rat_col])
                                    if rat_col in row.index and pd.notna(row[rat_col])
                                    else 0.0,
                                }
                            )
                    entry = {
                        "idx": int(idx),
                        "components": components,
                        "label": float(row["Experiment_value"]),
                    }
                    # Add NP ratio if available
                    if "IL_to_nucleicacid_massratio" in row.index and pd.notna(row["IL_to_nucleicacid_massratio"]):
                        entry["NP_ratio"] = float(row["IL_to_nucleicacid_massratio"])
                    entries.append(entry)
                formulations[label] = entries

            out_path = EXPORT_DIR / f"{sid}_s{seed}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "study_id": sid,
                        "pmid": si["pmid"],
                        "seed": seed,
                        "n_seed": len(seed_idx),
                        "n_pool": len(pool_idx),
                        "formulations": formulations,
                    },
                    f,
                    indent=2,
                )
            exported += 1

    print(f"Exported {exported} study x seed files to {EXPORT_DIR}")
    return exported


def import_comet_results(study_infos, seeds):
    """Import COMET predictions and compute benchmark metrics.

    Expects prediction files at:
      RESULTS_DIR/predictions/{study_id}_s{seed}_{mode}.json
    with format: {"predictions": [{"idx": int, "score": float}, ...]}
    """
    pred_dir = RESULTS_DIR / "predictions"
    if not pred_dir.exists():
        print(f"No predictions found at {pred_dir}")
        return

    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df

    for mode in ["zero_shot", "fine_tuned"]:
        print(f"\n--- Mode: {mode} ---")
        for si in study_infos:
            sid = get_study_id(si)
            study_df = df[df["Publication_PMID"] == float(si["pmid"])].copy()
            if len(study_df) == 0:
                continue

            batch_size = si.get("batch_size", BATCH_SIZE)
            n_rounds = si.get("n_rounds", MAX_ROUNDS)

            for seed in seeds:
                pred_path = pred_dir / f"{sid}_s{seed}_{mode}.json"
                if not pred_path.exists():
                    continue

                with open(pred_path) as f:
                    pred_data = json.load(f)

                # Reconstruct seed/pool split
                rng = np.random.RandomState(seed)
                n_seed = si["n_seed"]
                all_idx = list(study_df.index)
                rng.shuffle(all_idx)
                seed_idx = all_idx[:n_seed]
                pool_idx = all_idx[n_seed:]

                # Build score lookup from predictions
                score_lookup = {p["idx"]: p["score"] for p in pred_data["predictions"]}
                scores = np.array([score_lookup.get(idx, 0.0) for idx in pool_idx])

                # Compute top-k sets
                top_k_values = {}
                for pct, k in si["top_k_pct"].items():
                    actual_k = min(k, len(study_df))
                    top_k_values[pct] = set(study_df.nlargest(actual_k, "Experiment_value").index)

                # Rank and select
                total_budget = n_rounds * batch_size
                ranked = np.argsort(-scores)
                n_select = min(total_budget, len(pool_idx))
                selected = ranked[:n_select]

                training_idx = list(seed_idx)
                history = init_history(study_df, training_idx, top_k_values=top_k_values)

                for r in range(n_rounds):
                    start = r * batch_size
                    end = min((r + 1) * batch_size, n_select)
                    if start >= n_select:
                        break
                    batch_positions = selected[start:end]
                    batch_idx = [pool_idx[i] for i in batch_positions]
                    training_idx.extend(batch_idx)
                    update_history(history, study_df, training_idx, batch_idx, r, top_k_values=top_k_values)

                metrics = compute_metrics(history, top_k_values, len(study_df))
                metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

                recall_str = ", ".join(f"Top-{k}%={metrics['top_k_recall'].get(str(k), 0):.1%}" for k in [5, 10, 20])
                print(f"  {sid}/s{seed}: {recall_str}")

                result = {
                    "metrics": metrics,
                    "best_so_far": history["best_so_far"],
                    "round_best": history["round_best"],
                    "n_evaluated": history["n_evaluated"],
                }

                path = _per_seed_path(sid, mode, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                out = {
                    "baseline": f"comet_{mode}",
                    "study_id": sid,
                    "pmid": si["pmid"],
                    "seed": seed,
                    "study_info": {k: v for k, v in si.items() if k not in ("lnp_ids", "top_k_pct")},
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(path, "w") as f:
                    json.dump(out, f, indent=2, default=str)


def aggregate(study_infos, seeds):
    """Aggregate COMET results across studies and seeds."""
    if not RESULTS_DIR.exists():
        print("No results to aggregate.")
        return

    all_summaries = {}
    for mode in ["zero_shot", "fine_tuned"]:
        for si in study_infos:
            sid = get_study_id(si)
            vals = []
            for seed in seeds:
                path = _per_seed_path(sid, mode, seed)
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    recall = data["result"]["metrics"]["top_k_recall"]
                    vals.append(recall.get("5", 0.0))

            if vals:
                key = f"comet_{mode}"
                m = float(np.mean(vals))
                ci = bootstrap_ci(vals) if len(vals) >= 3 else (m, m)
                if key not in all_summaries:
                    display = "COMET Zero-Shot" if mode == "zero_shot" else "COMET Fine-Tuned"
                    all_summaries[key] = {"display": display, "studies": []}
                all_summaries[key]["studies"].append(
                    {
                        "study_id": sid,
                        "mean_top5": m,
                        "ci": ci,
                        "n_seeds": len(vals),
                    }
                )

    if all_summaries:
        print(f"\n{'Baseline':<25} {'Mean Top-5%':>12} {'95% CI':>18} {'N studies':>10}")
        print("-" * 70)
        for _key, data in sorted(all_summaries.items()):
            study_means = [s["mean_top5"] for s in data["studies"]]
            grand = float(np.mean(study_means))
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (grand, grand)
            print(f"{data['display']:<25} {grand:>11.1%} [{ci[0]:.1%}, {ci[1]:.1%}] {len(data['studies']):>10}")

        summary_path = RESULTS_DIR / "comet_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")


def run_local_inference(study_infos, seeds, comet_repo, weights_dir, nthreads=4, resume=False):
    """Run COMET inference locally using comet_infer.py.

    This calls comet_infer.py as a subprocess using the COMET venv.
    Requires: COMET venv at <comet_repo>/.venv with all dependencies.
    """
    import subprocess

    comet_repo = Path(comet_repo).resolve()
    comet_venv_python = comet_repo / ".venv" / "bin" / "python"
    infer_script = Path(__file__).resolve().parent / "comet_infer.py"

    if not comet_venv_python.exists():
        print(f"COMET venv not found at {comet_venv_python}")
        print("Set up the COMET environment first:")
        print(f"  cd {comet_repo}")
        print("  uv venv .venv --python 3.10")
        print(
            "  uv pip install torch rdkit-pypi lmdb numpy==1.24.4 "
            "pyprojroot pandas scikit-learn scipy ml-collections "
            "tensorboardX tqdm tokenizers"
        )
        return

    if not infer_script.exists():
        print(f"Inference script not found at {infer_script}")
        return

    exported_dir = str(EXPORT_DIR)
    output_dir = str(RESULTS_DIR / "predictions")

    # Filter studies/seeds to those that have exported data
    target_studies = ",".join(get_study_id(si) for si in study_infos)
    target_seeds = ",".join(str(s) for s in seeds)

    cmd = [
        str(comet_venv_python),
        str(infer_script),
        "--comet-repo",
        str(comet_repo),
        "--weights-dir",
        str(weights_dir),
        "--exported-dir",
        exported_dir,
        "--output-dir",
        output_dir,
        "--nthreads",
        str(nthreads),
        "--studies",
        target_studies,
        "--seeds",
        target_seeds,
    ]
    if resume:
        cmd.append("--resume")

    print("Running COMET inference...")
    print(f"  COMET repo: {comet_repo}")
    print(f"  Weights: {weights_dir}")
    print(f"  Python: {comet_venv_python}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent.parent))
    if result.returncode != 0:
        print(f"Inference failed with return code {result.returncode}")
    else:
        print("Inference completed successfully.")


def main():

    parser = argparse.ArgumentParser(description="COMET Baseline Wrapper")
    parser.add_argument("--export", action="store_true", help="Export study data for COMET inference")
    parser.add_argument("--run", action="store_true", help="Run COMET inference locally (CPU ok)")
    parser.add_argument("--import-results", action="store_true", help="Import COMET predictions and compute metrics")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing results")
    parser.add_argument("--comet-repo", type=str, default=None, help="Path to COMET repo (for --run)")
    parser.add_argument("--weights-dir", type=str, default=None, help="Path to COMET weights dir (for --run)")
    parser.add_argument("--nthreads", type=int, default=4, help="Threads for conformer generation (for --run)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed studies")
    parser.add_argument("--pmids", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--studies-json", type=str, default=None)
    args = parser.parse_args()

    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]

    # Load study infos
    if args.studies_json:
        with open(args.studies_json) as f:
            study_infos = json.load(f)
    else:
        default_studies = (
            Path(__file__).resolve().parent.parent.parent / "experiments" / "data_integrity" / "studies_with_ids.json"
        )
        if default_studies.exists():
            with open(default_studies) as f:
                study_infos = json.load(f)
        else:
            default_studies_alt = (
                Path(__file__).resolve().parent.parent.parent / "experiments" / "data_integrity" / "studies.json"
            )
            if default_studies_alt.exists():
                with open(default_studies_alt) as f:
                    study_infos = json.load(f)
            else:
                from LNPBO.data.lnpdb_bridge import load_lnpdb_full

                dataset = load_lnpdb_full()
                study_infos = characterize_studies(dataset.df)

    if args.pmids:
        target = set(args.pmids.split(","))
        study_infos = [
            si for si in study_infos if si.get("study_id", str(si["pmid"])) in target or str(si["pmid"]) in target
        ]

    ensure_top_k_pct(study_infos)

    if args.export:
        from LNPBO.data.lnpdb_bridge import load_lnpdb_full

        print("Loading LNPDB...")
        dataset = load_lnpdb_full()
        export_study_data(dataset.df, study_infos, seeds)

    elif args.run:
        if not args.comet_repo:
            print("--comet-repo is required for --run mode")
            return
        if not args.weights_dir:
            print("--weights-dir is required for --run mode")
            return
        # Ensure data is exported first
        if not EXPORT_DIR.exists() or not list(EXPORT_DIR.glob("*.json")):
            print("No exported data found. Running export first...")
            from LNPBO.data.lnpdb_bridge import load_lnpdb_full

            print("Loading LNPDB...")
            dataset = load_lnpdb_full()
            export_study_data(dataset.df, study_infos, seeds)

        run_local_inference(
            study_infos,
            seeds,
            args.comet_repo,
            args.weights_dir,
            nthreads=args.nthreads,
            resume=args.resume,
        )

    elif args.import_results:
        import_comet_results(study_infos, seeds)
        aggregate(study_infos, seeds)

    elif args.aggregate_only:
        aggregate(study_infos, seeds)

    else:
        print("Usage:")
        print("  --export          Export study data for COMET inference")
        print("  --run             Run COMET inference locally (CPU ok, no CUDA needed)")
        print("  --import-results  Import COMET predictions and compute metrics")
        print("  --aggregate-only  Only aggregate existing results")
        print()
        print("Full local workflow (single machine, CPU ok):")
        print("  python -m benchmarks.baselines.comet_wrapper --export")
        print("  python -m benchmarks.baselines.comet_wrapper --run \\")
        print("      --comet-repo /path/to/COMET \\")
        print("      --weights-dir /path/to/COMET/experiments/weights/<model_dir>")
        print("  python -m benchmarks.baselines.comet_wrapper --import-results")
        print()
        print("Two-machine workflow:")
        print("  1. Run --export on any machine")
        print("  2. Transfer exported_data/ to a machine with COMET")
        print("  3. Run comet_infer.py (works on CPU or CUDA)")
        print("  4. Transfer predictions/ back")
        print("  5. Run --import-results to compute metrics")


if __name__ == "__main__":
    main()
