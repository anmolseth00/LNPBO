#!/usr/bin/env python3
"""Direct COMET inference on exported LNPBO study data.

This script runs inside the COMET repo's Python environment and performs
zero-shot inference on LNPDB formulations exported by comet_wrapper.py.

It bypasses the full COMET training pipeline by directly constructing the
LMDB datasets that COMET expects, loading pretrained weights, and running
the forward pass on CPU.

Usage (from LNPBO repo root):

    # 1. Export study data (uses LNPBO venv)
    python -m benchmarks.baselines.comet_wrapper --export

    # 2. Run inference (uses COMET venv, can run on CPU)
    /path/to/COMET/.venv/bin/python benchmarks/baselines/comet_infer.py \
        --comet-repo /path/to/COMET \
        --weights-dir /path/to/COMET/experiments/weights/weights \
        --exported-dir benchmark_results/baselines/comet/exported_data \
        --output-dir benchmark_results/baselines/comet/predictions

    # 3. Import results and compute metrics (uses LNPBO venv)
    python -m benchmarks.baselines.comet_wrapper --import-results

Prerequisites:
    - COMET repo with pretrained weights downloaded
    - COMET venv with: torch, rdkit, lmdb, numpy<2, pyprojroot, pandas, scikit-learn
    - Exported JSON files from step 1

The COMET model (NPUniMolModel) is a multi-task architecture trained on
DC2.4 and B16F10 cell lines with contrastive loss. For zero-shot ranking
we use the DC2.4 prediction head (in_house_lnp_DC24_luc). If both heads
are available we average them.

Reference: Chang et al., "COMET: Composite Material Transformer for
Lipid Nanoparticle Design", NeurIPS 2024 ML4H Workshop.
"""

import argparse
import contextlib
import copy
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# PyTorch 2.x compatibility: patch torch.load to use weights_only=False
# COMET checkpoints contain argparse.Namespace objects that require pickle.
# ---------------------------------------------------------------------------
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


# ---------------------------------------------------------------------------
# Conformer generation (mirrors COMET's preprocess_data_LANCE.ipynb)
# ---------------------------------------------------------------------------


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    return coordinates


def smi2_3Dcoords(smi, cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except Exception:
                    coordinates = smi2_2Dcoords(smi)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except Exception:
                    coordinates = smi2_2Dcoords(smi)
            else:
                coordinates = smi2_2Dcoords(smi)
        except Exception:
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape mismatch for {smi}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smi_to_mol_data(smi, conf_size=10):
    """Convert SMILES to mol data dict (atoms, coordinates, smi)."""
    cnt = conf_size
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if len(mol.GetAtoms()) > 400:
        coordinate_list = [smi2_2Dcoords(smi)] * (cnt + 1)
    else:
        coordinate_list = smi2_3Dcoords(smi, cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return {"atoms": atoms, "coordinates": coordinate_list, "smi": smi}


def _smi_to_mol_data_pickled(smi):
    """Wrapper for multiprocessing: returns pickled mol data."""
    try:
        data = smi_to_mol_data(smi)
        if data is not None:
            return pickle.dumps(data, protocol=-1)
    except Exception as e:
        print(f"Failed SMILES: {smi}: {e}")
    return None


# ---------------------------------------------------------------------------
# LMDB construction
# ---------------------------------------------------------------------------


def build_mol_lmdb(smi_list, output_path, nthreads=4):
    """Build mol.lmdb from a list of unique SMILES.

    Returns (n_mols, smi2mol_id) where smi2mol_id maps each successfully
    processed SMILES to its index in the LMDB.
    """
    with contextlib.suppress(OSError):
        os.remove(output_path)

    env = lmdb.open(
        output_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn = env.begin(write=True)
    smi2mol_id = {}
    with Pool(nthreads) as pool:
        i = 0
        for smi, result in tqdm(
            zip(smi_list, pool.imap(_smi_to_mol_data_pickled, smi_list)),
            total=len(smi_list),
            desc="Building mol.lmdb",
        ):
            if result is not None:
                txn.put(f"{i}".encode("ascii"), result)
                smi2mol_id[smi] = i
                i += 1
    txn.commit()
    env.close()
    return i, smi2mol_id


def build_infer_lmdb(formulations, smi2mol_id, output_path):
    """Build infer.lmdb from formulation entries.

    Each entry has: mol_id (list), percent (list), component_type (list),
    target (dict), dataset_name, components, lnp_id
    """
    with contextlib.suppress(OSError):
        os.remove(output_path)

    env = lmdb.open(
        output_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn = env.begin(write=True)

    for i, entry in enumerate(formulations):
        mol_ids = []
        percents = []
        component_types = []
        components_out = []

        for comp in entry["components"]:
            smi = comp["smi"]
            if smi not in smi2mol_id:
                continue
            mol_id = smi2mol_id[smi]
            mol_ids.append(mol_id)
            percents.append(comp["mol"])
            component_types.append(comp["component_type"])
            comp_out = copy.deepcopy(comp)
            comp_out["mol_id"] = mol_id
            comp_out["percent"] = comp["mol"]
            components_out.append(comp_out)

        # Both task labels are needed since the model has two classification heads.
        # Use the exported label for both (dummy value for inference -- only
        # the predict output matters, not the loss).
        label_val = entry.get("label", 0.0)
        labels = {
            "in_house_lnp_DC24_luc": label_val,
            "in_house_lnp_B16F10_luc": label_val,
        }

        data = {
            "mol_id": mol_ids,
            "percent": percents,
            "component_type": component_types,
            "target": labels,
            "dataset_name": "in_house_lnp",
            "components": components_out,
            "lnp_id": str(entry.get("idx", i)),
        }
        # Add NP_ratio if available
        if "NP_ratio" in entry:
            data["NP_ratio"] = entry["NP_ratio"]

        txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))

    txn.commit()
    env.close()
    return len(formulations)


# ---------------------------------------------------------------------------
# COMET inference
# ---------------------------------------------------------------------------


def run_comet_inference(
    comet_repo,
    weights_path,
    task_schema_path,
    data_dir,
    task_name,
    batch_size=64,
    conf_size=11,
):
    """Run COMET model inference and return predictions.

    Parameters
    ----------
    comet_repo : str
        Path to the COMET repository root.
    weights_path : str
        Path to checkpoint_best.pt.
    task_schema_path : str
        Path to task schema JSON (relative to data_dir).
    data_dir : str
        Directory containing dict.txt and the task subdirectory.
    task_name : str
        Name of the task directory (contains subdatasets with infer.lmdb).
    batch_size : int
        Inference batch size.
    conf_size : int
        Number of conformers (should match training: 11).

    Returns
    -------
    predictions : dict
        Mapping from lnp_id (str) to predicted score (float).
    reduced : dict
        Full reduced metrics dict from COMET.
    """
    # Add COMET to path
    if comet_repo not in sys.path:
        sys.path.insert(0, comet_repo)
    import importlib

    importlib.import_module("unimol")

    from unimol.core import checkpoint_utils, options, utils
    from unimol.core import tasks as core_tasks
    from unimol.core.logging import progress_bar

    # Build args manually (mirrors inference_script_LANCE_lipid_I01.py)
    parser = options.get_validation_parser()
    options.add_model_args(parser)

    args_list = [
        data_dir,
        "--task-name",
        task_name,
        "--valid-subset",
        "infer",
        "--num-workers",
        "0",
        "--ddp-backend=c10d",
        "--batch-size",
        str(batch_size),
        "--task",
        "mol_np_finetune",
        "--loss",
        "np_finetune_contrastive",
        "--arch",
        "np_unimol",
        "--classification-head-name",
        task_name,
        "--num-classes",
        "1",
        "--dict-name",
        "dict.txt",
        "--conf-size",
        str(conf_size),
        "--only-polar",
        "0",
        "--path",
        weights_path,
        "--log-interval",
        "50",
        "--log-format",
        "simple",
        "--results-path",
        os.path.join(data_dir, "results"),
        "--lnp-encoder-layers",
        "8",
        "--lnp-encoder-embed-dim",
        "256",
        "--lnp-encoder-ffn-embed-dim",
        "256",
        "--lnp-encoder-attention-heads",
        "8",
        "--full-dataset-task-schema-path",
        task_schema_path,
        "--load-full-np-model",
        "--concat-datasets",
        "--output-cls-rep",
        "--cpu",
    ]

    args = options.parse_args_and_arch(parser, input_args=args_list)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load model
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = core_tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    if use_cuda:
        model.cuda()
    model.eval()

    # Build loss
    loss = task.build_loss(args)
    loss.eval()

    # Load dataset
    task.load_concat_dataset("infer", combine=False, epoch=1)
    dataset = task.dataset("infer")
    print(f"Inference dataset size: {len(dataset)}")

    # Build iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        prefix="inference",
        default_log_format="simple",
    )

    log_outputs = []
    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if len(sample) == 0:
            continue
        _, _, log_output = task.valid_step(sample, model, loss, test=True, infer=True, output_cls_rep=True)
        progress.log({}, step=i)
        log_outputs.append(log_output)

    # Reduce and extract predictions
    reduced = task.reduce_metrics(log_outputs, loss, "infer", infer=True)

    # Extract predictions from reduced metrics.
    # The contrastive loss produces keys like:
    #   predict_{task_name}  -> per-log-output
    #   {task_name}infer_predict  -> aggregated in reduced_metrics_dict
    # And lnp_ids are collected as reduced['lnp_ids'].
    lnp_ids = reduced.get("lnp_ids", [])

    # Try to get predictions from DC2.4 head (primary), fall back to B16F10
    pred_keys = [k for k in reduced if "infer_predict" in k]
    print(f"Available prediction keys: {pred_keys}")
    print(f"All reduced keys: {list(reduced.keys())}")

    pred_tensors = {}
    for k in pred_keys:
        v = reduced[k]
        if hasattr(v, "numpy"):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            pred_tensors[k] = v.flatten()
            print(f"  {k}: shape={v.shape}")

    # Prefer DC24_luc, then B16F10_luc, then average, then any available
    dc24_key = [k for k in pred_tensors if "DC24" in k]
    b16_key = [k for k in pred_tensors if "B16F10" in k]

    if dc24_key and b16_key:
        # Average both heads
        scores = (pred_tensors[dc24_key[0]] + pred_tensors[b16_key[0]]) / 2.0
        print("Using average of DC24 and B16F10 predictions")
    elif dc24_key:
        scores = pred_tensors[dc24_key[0]]
        print("Using DC24 predictions only")
    elif b16_key:
        scores = pred_tensors[b16_key[0]]
        print("Using B16F10 predictions only")
    elif pred_tensors:
        # Use whatever is available
        first_key = next(iter(pred_tensors.keys()))
        scores = pred_tensors[first_key]
        print(f"Using {first_key} predictions")
    else:
        print("WARNING: No prediction tensors found in reduced metrics!")
        scores = np.zeros(len(lnp_ids))

    # Build predictions dict mapping lnp_id -> score
    predictions = {}
    for idx, lnp_id in enumerate(lnp_ids):
        if idx < len(scores):
            predictions[str(lnp_id)] = float(scores[idx])

    if predictions:
        print(
            f"Extracted {len(predictions)} predictions, "
            f"score range: [{min(predictions.values()):.4f}, {max(predictions.values()):.4f}]"
        )
    else:
        print("WARNING: No predictions extracted!")

    return predictions, reduced


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def process_study_file(
    exported_path,
    comet_repo,
    weights_dir,
    output_dir,
    nthreads=4,
):
    """Process a single exported study JSON file.

    Creates temporary LMDB datasets, runs inference, and saves predictions.
    """
    with open(exported_path) as f:
        data = json.load(f)

    study_id = data["study_id"]
    seed = data["seed"]
    pool_entries = data["formulations"]["pool"]

    # Collect unique SMILES
    unique_smiles = []
    for entry in pool_entries:
        for comp in entry["components"]:
            if comp["smi"] not in unique_smiles:
                unique_smiles.append(comp["smi"])

    print(f"\n{'=' * 60}")
    print(f"Study: {study_id}, Seed: {seed}")
    print(f"Pool: {len(pool_entries)}, Unique SMILES: {len(unique_smiles)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy necessary files from COMET experiments dir
        comet_experiments = os.path.join(comet_repo, "experiments")

        # Copy dict.txt
        dict_src = os.path.join(comet_experiments, "dict.txt")
        dict_dst = os.path.join(tmpdir, "dict.txt")
        if os.path.exists(dict_src):
            shutil.copy(dict_src, dict_dst)
        else:
            raise FileNotFoundError(f"dict.txt not found at {dict_src}")

        # Copy task schema
        schema_name = "in_house_lnp_master_schema_NPratio_AOvolratio.json"
        task_schemas_dir = os.path.join(tmpdir, "task_schemas")
        os.makedirs(task_schemas_dir, exist_ok=True)
        schema_src = os.path.join(comet_experiments, "task_schemas", schema_name)
        schema_dst = os.path.join(task_schemas_dir, schema_name)
        if os.path.exists(schema_src):
            shutil.copy(schema_src, schema_dst)
        else:
            raise FileNotFoundError(f"Task schema not found at {schema_src}")

        # Directory structure COMET expects:
        #   <data_dir>/dict.txt
        #   <data_dir>/task_schemas/<schema>.json
        #   <data_dir>/<task_name>/mol.lmdb
        #   <data_dir>/<task_name>/<subdataset_name>/infer.lmdb
        task_name = "lnpbo_study"
        subdataset_name = "in_house_lnp"
        task_dir = os.path.join(tmpdir, task_name)
        subdataset_dir = os.path.join(task_dir, subdataset_name)
        os.makedirs(subdataset_dir, exist_ok=True)

        # Build mol.lmdb from unique SMILES
        mol_lmdb_path = os.path.join(task_dir, "mol.lmdb")
        n_mols, smi2mol_id = build_mol_lmdb(unique_smiles, mol_lmdb_path, nthreads=nthreads)
        print(f"Built mol.lmdb: {n_mols} molecules")

        # Build infer.lmdb
        infer_lmdb_path = os.path.join(subdataset_dir, "infer.lmdb")
        n_infer = build_infer_lmdb(pool_entries, smi2mol_id, infer_lmdb_path)
        print(f"Built infer.lmdb: {n_infer} formulations")

        # Find weights checkpoint
        weights_path = None
        if os.path.isfile(weights_dir):
            weights_path = weights_dir
        elif os.path.isdir(weights_dir):
            candidate = os.path.join(weights_dir, "checkpoint_best.pt")
            if os.path.exists(candidate):
                weights_path = candidate
            else:
                for root, _dirs, files in os.walk(weights_dir):
                    for f in files:
                        if f == "checkpoint_best.pt":
                            weights_path = os.path.join(root, f)
                            break
                    if weights_path:
                        break

        if weights_path is None:
            raise FileNotFoundError(
                f"No checkpoint_best.pt found in {weights_dir}. "
                "Download weights from: "
                "https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv"
            )
        print(f"Using weights: {weights_path}")

        # Run inference
        t0 = time.time()
        predictions, _reduced = run_comet_inference(
            comet_repo=comet_repo,
            weights_path=weights_path,
            task_schema_path=os.path.join("task_schemas", schema_name),
            data_dir=tmpdir,
            task_name=task_name,
        )
        elapsed = time.time() - t0
        print(f"Inference completed in {elapsed:.1f}s")

        # Map predictions back to original indices
        pred_list = []
        for entry in pool_entries:
            lnp_id = str(entry["idx"])
            score = predictions.get(lnp_id, 0.0)
            pred_list.append({"idx": entry["idx"], "score": score})

        # Save predictions
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{study_id}_s{seed}_zero_shot.json")
        with open(out_path, "w") as f:
            json.dump({"predictions": pred_list}, f, indent=2)
        print(f"Saved predictions to {out_path}")

    return out_path


def process_study_file_ensemble(
    exported_path,
    comet_repo,
    weights_dirs,
    output_dir,
    nthreads=4,
):
    """Process a single study with an ensemble of COMET folds.

    Runs inference with each fold's weights, then averages the predictions.
    """
    with open(exported_path) as f:
        data = json.load(f)

    study_id = data["study_id"]
    seed = data["seed"]
    pool_entries = data["formulations"]["pool"]

    # Collect unique SMILES
    unique_smiles = []
    for entry in pool_entries:
        for comp in entry["components"]:
            if comp["smi"] not in unique_smiles:
                unique_smiles.append(comp["smi"])

    print(f"\n{'=' * 60}")
    print(f"Study: {study_id}, Seed: {seed}")
    print(f"Pool: {len(pool_entries)}, Unique SMILES: {len(unique_smiles)}")
    print(f"Ensemble: {len(weights_dirs)} folds")

    with tempfile.TemporaryDirectory() as tmpdir:
        comet_experiments = os.path.join(comet_repo, "experiments")

        # Copy dict.txt
        shutil.copy(
            os.path.join(comet_experiments, "dict.txt"),
            os.path.join(tmpdir, "dict.txt"),
        )

        # Copy task schema
        schema_name = "in_house_lnp_master_schema_NPratio_AOvolratio.json"
        task_schemas_dir = os.path.join(tmpdir, "task_schemas")
        os.makedirs(task_schemas_dir, exist_ok=True)
        shutil.copy(
            os.path.join(comet_experiments, "task_schemas", schema_name),
            os.path.join(task_schemas_dir, schema_name),
        )

        # Build data dirs
        task_name = "lnpbo_study"
        subdataset_name = "in_house_lnp"
        task_dir = os.path.join(tmpdir, task_name)
        subdataset_dir = os.path.join(task_dir, subdataset_name)
        os.makedirs(subdataset_dir, exist_ok=True)

        # Build mol.lmdb (once for all folds)
        mol_lmdb_path = os.path.join(task_dir, "mol.lmdb")
        n_mols, smi2mol_id = build_mol_lmdb(unique_smiles, mol_lmdb_path, nthreads=nthreads)
        print(f"Built mol.lmdb: {n_mols} molecules")

        # Build infer.lmdb
        infer_lmdb_path = os.path.join(subdataset_dir, "infer.lmdb")
        build_infer_lmdb(pool_entries, smi2mol_id, infer_lmdb_path)

        # Run inference with each fold
        all_scores = {}  # lnp_id -> list of scores
        for fold_i, wd in enumerate(weights_dirs):
            weights_path = os.path.join(wd, "checkpoint_best.pt")
            if not os.path.exists(weights_path):
                print(f"  Fold {fold_i}: weights not found at {weights_path}, skipping")
                continue
            print(f"\n--- Fold {fold_i}: {os.path.basename(wd)} ---")

            t0 = time.time()
            predictions, _ = run_comet_inference(
                comet_repo=comet_repo,
                weights_path=weights_path,
                task_schema_path=os.path.join("task_schemas", schema_name),
                data_dir=tmpdir,
                task_name=task_name,
            )
            elapsed = time.time() - t0
            print(f"  Fold {fold_i} inference: {elapsed:.1f}s, {len(predictions)} predictions")

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
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{study_id}_s{seed}_zero_shot.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "predictions": pred_list,
                    "n_folds": len(weights_dirs),
                    "n_folds_used": sum(
                        1 for wd in weights_dirs if os.path.exists(os.path.join(wd, "checkpoint_best.pt"))
                    ),
                },
                f,
                indent=2,
            )
        print(f"\nSaved ensemble predictions to {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="COMET direct inference on exported LNPBO study data")
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
        help="Path to COMET weights directory (containing checkpoint_best.pt). "
        "If this contains multiple fold subdirs, ensemble mode is used.",
    )
    parser.add_argument(
        "--exported-dir",
        type=str,
        default="benchmark_results/baselines/comet/exported_data",
        help="Directory with exported study JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/baselines/comet/predictions",
        help="Directory for prediction output files",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=4,
        help="Number of threads for conformer generation",
    )
    parser.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Comma-separated study IDs to process (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to process (default: all found)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip studies that already have predictions",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use all fold subdirectories under --weights-dir as an ensemble",
    )
    args = parser.parse_args()

    exported_dir = Path(args.exported_dir)
    if not exported_dir.exists():
        print(f"Exported data directory not found: {exported_dir}")
        print("Run 'python -m benchmarks.baselines.comet_wrapper --export' first.")
        sys.exit(1)

    # Find all exported files
    exported_files = sorted(exported_dir.glob("*.json"))
    if not exported_files:
        print(f"No exported JSON files found in {exported_dir}")
        sys.exit(1)

    # Filter by study/seed if specified
    if args.studies:
        target_studies = set(args.studies.split(","))
        exported_files = [f for f in exported_files if any(f.stem.startswith(s) for s in target_studies)]
    if args.seeds:
        target_seeds = set(f"s{s}" for s in args.seeds.split(","))
        exported_files = [f for f in exported_files if any(f.stem.endswith(s) for s in target_seeds)]

    # Filter already-processed if resuming
    output_dir = Path(args.output_dir)
    if args.resume:
        remaining = []
        for f in exported_files:
            stem = f.stem
            pred_path = output_dir / f"{stem}_zero_shot.json"
            if not pred_path.exists():
                remaining.append(f)
        exported_files = remaining

    print(f"Processing {len(exported_files)} exported files")
    print(f"COMET repo: {args.comet_repo}")
    print(f"Weights: {args.weights_dir}")

    # Determine if ensemble mode
    weights_dir = Path(args.weights_dir)
    ensemble_dirs = None
    if args.ensemble or not (weights_dir / "checkpoint_best.pt").exists():
        # Look for fold subdirectories
        fold_dirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and (d / "checkpoint_best.pt").exists()])
        # Only use lipid LNP folds (fig3d), not PBAE (fig4c)
        lipid_dirs = [d for d in fold_dirs if "fig3d" in d.name]
        if lipid_dirs:
            ensemble_dirs = [str(d) for d in lipid_dirs]
            print(f"Ensemble mode: {len(ensemble_dirs)} lipid LNP folds")
        elif fold_dirs:
            ensemble_dirs = [str(d) for d in fold_dirs]
            print(f"Ensemble mode: {len(ensemble_dirs)} folds (all)")

    for i, export_path in enumerate(exported_files):
        print(f"\n[{i + 1}/{len(exported_files)}] {export_path.name}")
        try:
            if ensemble_dirs:
                process_study_file_ensemble(
                    exported_path=str(export_path),
                    comet_repo=args.comet_repo,
                    weights_dirs=ensemble_dirs,
                    output_dir=args.output_dir,
                    nthreads=args.nthreads,
                )
            else:
                process_study_file(
                    exported_path=str(export_path),
                    comet_repo=args.comet_repo,
                    weights_dir=args.weights_dir,
                    output_dir=args.output_dir,
                    nthreads=args.nthreads,
                )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
