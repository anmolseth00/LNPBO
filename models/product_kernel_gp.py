#!/usr/bin/env python3
"""Product kernel GP: Tanimoto on molecular fingerprints * Matern-ARD on ratios.

SHAP attribution shows 82% of signal is IL molecular identity and 14% is
molar ratios. A single isotropic kernel conflates these subspaces. The
product kernel decomposes the problem:

    k(x, y) = k_mol(x_mol, y_mol) * k_ratio(x_ratio, y_ratio)

where k_mol is the Tanimoto kernel on raw count Morgan fingerprints (2048-bit)
and k_ratio is a Matern-5/2 kernel with ARD lengthscales on molar ratios.

The ARD lengthscales learn which ratio dimensions are relevant. An additive
variant (k_mol + k_ratio) is also tested for comparison.

Raw molar ratios are used
as a fallback. The 4 molar ratios are: IL_molratio, HL_molratio, CHL_molratio,
PEG_molratio (summing to ~100).

References:
    Ralaivola, L. et al. (2005). "Graph Kernels for Chemical Informatics."
        Neural Networks 18(8), 1093-1110.
    Griffiths, R.-R. et al. (2023). "GAUCHE: A Library for Gaussian Processes
        in Chemistry." NeurIPS 2023.
    Tripp, A. et al. (2023). "Tanimoto Random Features for Scalable Molecular
        Machine Learning." NeurIPS 2023.
"""

import json
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from LNPBO.data.generate_morgan_fingerprints import single_morgan_fingerprints
from LNPBO.diagnostics.utils import load_lnpdb_clean, study_split
from LNPBO.models.splits import scaffold_split
from LNPBO.models.tanimoto_gp import TanimotoKernel


RATIO_COLS = ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]


def _try_ilr_transform(ratios):
    """Attempt ILR transform; fall back to raw ratios if unavailable.

    The ILR (Isometric Log-Ratio) transform maps D simplex components to
    D-1 unconstrained coordinates. Falls back to raw ratios if unavailable.

    Reference: Egozcue, J.J. et al. (2003). "Isometric Logratio
    Transformations for Compositional Data Analysis." Math. Geol. 35(3).
    """
    try:
        from LNPBO.data.compositional import ilr_transform
        return ilr_transform(ratios), "ilr"
    except (ImportError, ModuleNotFoundError):
        return ratios, "raw"


class ProductKernelSVGP(gpytorch.models.ApproximateGP):
    """SVGP with product kernel: Tanimoto(mol) * Matern-ARD(ratio).

    The product kernel captures the interaction structure: the effect of
    molar ratios depends on which molecule is used (and vice versa).
    """

    def __init__(self, inducing_points, n_mol_dims, n_ratio_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        mol_dims = list(range(n_mol_dims))
        ratio_dims = list(range(n_mol_dims, n_mol_dims + n_ratio_dims))

        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoKernel(active_dims=mol_dims)
            * gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=n_ratio_dims,
                active_dims=ratio_dims,
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class AdditiveKernelSVGP(gpytorch.models.ApproximateGP):
    """SVGP with additive kernel: Tanimoto(mol) + Matern-ARD(ratio).

    The additive kernel is less restrictive than the product kernel:
    it assumes the molecular and ratio contributions are independent.
    """

    def __init__(self, inducing_points, n_mol_dims, n_ratio_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        mol_dims = list(range(n_mol_dims))
        ratio_dims = list(range(n_mol_dims, n_mol_dims + n_ratio_dims))

        self.k_mol = gpytorch.kernels.ScaleKernel(
            TanimotoKernel(active_dims=mol_dims)
        )
        self.k_ratio = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=n_ratio_dims,
                active_dims=ratio_dims,
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.k_mol(x) + self.k_ratio(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TanimotoOnlySVGP(gpytorch.models.ApproximateGP):
    """SVGP with Tanimoto-only kernel (molecular features, no ratios)."""

    def __init__(self, inducing_points, n_mol_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        mol_dims = list(range(n_mol_dims))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoKernel(active_dims=mol_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MaternOnlySVGP(gpytorch.models.ApproximateGP):
    """SVGP with Matern-ARD on ratio features only (no molecular features)."""

    def __init__(self, inducing_points, n_mol_dims, n_ratio_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        ratio_dims = list(range(n_mol_dims, n_mol_dims + n_ratio_dims))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=n_ratio_dims,
                active_dims=ratio_dims,
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _get_raw_il_fingerprints(df, n_bits=2048, radius=3):
    """Get raw count Morgan fingerprints for each row via IL_SMILES."""
    unique_smiles = df["IL_SMILES"].dropna().unique().tolist()
    fp_map = {}
    for smi in unique_smiles:
        fp = single_morgan_fingerprints(smi, radius=radius, n_bits=n_bits, count=True)
        fp_map[smi] = fp

    fps = np.array([
        fp_map.get(smi, np.zeros(n_bits))
        for smi in df["IL_SMILES"].values
    ], dtype=np.float32)
    return fps


def _get_ratio_features(df):
    """Extract molar ratio features and apply ILR if available."""
    ratios = df[RATIO_COLS].values.astype(np.float32)
    ratios_transformed, transform_type = _try_ilr_transform(ratios)
    return ratios_transformed.astype(np.float32), transform_type


def _select_diverse_inducing(train_x, n_inducing, seed=42):
    """Select inducing points from unique rows to avoid singularity."""
    unique_x, inverse = torch.unique(train_x, dim=0, return_inverse=True)
    m = min(n_inducing, unique_x.size(0))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(unique_x.size(0), generator=rng)
    return unique_x[perm[:m]].clone(), m


def _train_svgp(model, likelihood, train_x, train_y, epochs=50, batch_size=1024, lr=0.01):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    with gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6), \
         gpytorch.settings.cholesky_max_tries(6):
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                output = model(xb)
                loss = -mll(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, loss={epoch_loss / n_batches:.4f}")


def _predict(model, likelihood, test_x, batch_size=2048):
    model.eval()
    likelihood.eval()
    all_mu = []
    all_sigma = []
    with torch.no_grad(), \
         gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6), \
         gpytorch.settings.cholesky_max_tries(6):
        for i in range(0, test_x.size(0), batch_size):
            xb = test_x[i:i + batch_size]
            pred = likelihood(model(xb))
            all_mu.append(pred.mean.cpu())
            all_sigma.append(pred.stddev.cpu())
    return torch.cat(all_mu).numpy(), torch.cat(all_sigma).numpy()


def _ece(y_true, mu, sigma, alpha=0.1):
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    ece = abs(coverage - (1 - alpha))
    return coverage, ece


def _extract_ard_lengthscales(model, kernel_type):
    """Extract learned ARD lengthscales from the Matern kernel."""
    try:
        if kernel_type == "product":
            # ScaleKernel(Tanimoto * Matern) -- ProductKernel has sub-kernels
            product = model.covar_module.base_kernel
            matern = product.kernels[1]
            ls = matern.lengthscale.detach().cpu().numpy().flatten()
        elif kernel_type == "additive":
            matern = model.k_ratio.base_kernel
            ls = matern.lengthscale.detach().cpu().numpy().flatten()
        elif kernel_type == "matern_only":
            matern = model.covar_module.base_kernel
            ls = matern.lengthscale.detach().cpu().numpy().flatten()
        else:
            return None
        return {col: round(float(v), 4) for col, v in zip(RATIO_COLS, ls)}
    except Exception as e:
        print(f"    Warning: could not extract lengthscales: {e}")
        return None


def _run_kernel_comparison(
    X_mol_train, X_mol_test, X_ratio_train, X_ratio_test,
    y_train, y_test, split_name,
    n_inducing=512, epochs=50, batch_size=1024, seed=42, ratio_transform="raw",
):
    """Run all kernel variants on a given split."""
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 0 else 1.0
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    # Standardize ratios (important for Matern ARD)
    ratio_scaler = StandardScaler().fit(X_ratio_train)
    X_ratio_train_s = ratio_scaler.transform(X_ratio_train).astype(np.float32)
    X_ratio_test_s = ratio_scaler.transform(X_ratio_test).astype(np.float32)

    n_mol = X_mol_train.shape[1]
    n_ratio = X_ratio_train_s.shape[1]

    # Concatenate: [mol_features | ratio_features]
    X_train_cat = np.hstack([X_mol_train, X_ratio_train_s])
    X_test_cat = np.hstack([X_mol_test, X_ratio_test_s])

    train_x = torch.tensor(X_train_cat, dtype=torch.float32)
    train_y = torch.tensor(y_train_s, dtype=torch.float32)
    test_x = torch.tensor(X_test_cat, dtype=torch.float32)

    results = {}

    kernels_to_test = {
        "product": lambda ip: ProductKernelSVGP(ip, n_mol, n_ratio),
        "additive": lambda ip: AdditiveKernelSVGP(ip, n_mol, n_ratio),
        "tanimoto_only": lambda ip: TanimotoOnlySVGP(ip, n_mol),
        "matern_only": lambda ip: MaternOnlySVGP(ip, n_mol, n_ratio),
    }

    for kernel_name, model_fn in kernels_to_test.items():
        print(f"\n  [{split_name}] Training {kernel_name} SVGP...")
        inducing_points, m = _select_diverse_inducing(train_x, n_inducing, seed=seed)

        model = model_fn(inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        t0 = time.time()
        _train_svgp(model, likelihood, train_x, train_y, epochs=epochs, batch_size=batch_size)
        train_time = time.time() - t0

        mu, sigma = _predict(model, likelihood, test_x)
        r2 = float(r2_score(y_test_s, mu))
        cov, ece = _ece(y_test_s, mu, sigma)
        width = float(2 * norm.ppf(0.95) * np.mean(sigma))

        ard_ls = _extract_ard_lengthscales(model, kernel_name)

        result_entry = {
            "r2": r2,
            "coverage_90": cov,
            "ece_90": ece,
            "mean_interval_width": width,
            "n_mol_dims": n_mol,
            "n_ratio_dims": n_ratio,
            "ratio_transform": ratio_transform,
            "n_inducing": m,
            "train_time_s": round(train_time, 1),
        }
        if ard_ls is not None:
            result_entry["ard_lengthscales"] = ard_ls

        results[kernel_name] = result_entry
        print(f"  [{split_name}] {kernel_name}: R^2={r2:.4f}, coverage={cov:.3f}, time={train_time:.1f}s")
        if ard_ls is not None:
            print(f"  [{split_name}] {kernel_name} ARD lengthscales: {ard_ls}")

    return results


def main() -> int:
    SEEDS = [42, 123, 456, 789, 2024]
    N_BITS = 2048
    RADIUS = 3
    N_INDUCING = 512
    EPOCHS = 30
    BATCH_SIZE = 1024

    print("Loading LNPDB clean data...")
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Ensure ratio columns exist and are complete
    for col in RATIO_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=RATIO_COLS).reset_index(drop=True)

    print(f"Computing raw count Morgan fingerprints ({N_BITS}-bit, radius={RADIUS})...")
    X_fps = _get_raw_il_fingerprints(df, n_bits=N_BITS, radius=RADIUS)

    print("Extracting ratio features...")
    X_ratios, ratio_transform = _get_ratio_features(df)
    print(f"  Ratio transform: {ratio_transform}, shape: {X_ratios.shape}")

    y = df["Experiment_value"].values.astype(np.float32)
    il_smiles = df["IL_SMILES"].tolist()

    all_results = {"config": {
        "n_bits": N_BITS,
        "radius": RADIUS,
        "n_inducing": N_INDUCING,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seeds": SEEDS,
        "ratio_transform": ratio_transform,
        "ratio_columns": RATIO_COLS,
        "n_mol_dims": N_BITS,
        "n_ratio_dims": X_ratios.shape[1],
    }}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        seed_results = {}

        # --- Scaffold split ---
        train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=seed)
        train_idx = sorted(set(train_idx + val_idx))

        seed_results["scaffold_split"] = _run_kernel_comparison(
            X_fps[train_idx], X_fps[test_idx],
            X_ratios[train_idx], X_ratios[test_idx],
            y[train_idx], y[test_idx],
            split_name="scaffold",
            n_inducing=N_INDUCING, epochs=EPOCHS, batch_size=BATCH_SIZE,
            seed=seed, ratio_transform=ratio_transform,
        )

        # --- Study-level split ---
        train_ids, test_ids = study_split(df, seed=seed)
        train_mask = df["study_id"].isin(train_ids).values
        test_mask = df["study_id"].isin(test_ids).values

        seed_results["study_split"] = _run_kernel_comparison(
            X_fps[train_mask], X_fps[test_mask],
            X_ratios[train_mask], X_ratios[test_mask],
            y[train_mask], y[test_mask],
            split_name="study",
            n_inducing=N_INDUCING, epochs=EPOCHS, batch_size=BATCH_SIZE,
            seed=seed, ratio_transform=ratio_transform,
        )

        all_results[f"seed_{seed}"] = seed_results

    # Aggregate across seeds
    kernel_types = ["product", "additive", "tanimoto_only", "matern_only"]
    for split_type in ["scaffold_split", "study_split"]:
        for kernel_type in kernel_types:
            r2s = [
                all_results[f"seed_{s}"][split_type][kernel_type]["r2"]
                for s in SEEDS
            ]
            summary_entry = {
                "r2_mean": round(float(np.mean(r2s)), 4),
                "r2_std": round(float(np.std(r2s)), 4),
                "r2_values": [round(v, 4) for v in r2s],
            }

            # Aggregate ARD lengthscales across seeds
            all_ls = [
                all_results[f"seed_{s}"][split_type][kernel_type].get("ard_lengthscales")
                for s in SEEDS
            ]
            all_ls = [ls for ls in all_ls if ls is not None]
            if all_ls:
                mean_ls = {}
                for col in RATIO_COLS:
                    vals = [ls[col] for ls in all_ls if col in ls]
                    if vals:
                        mean_ls[col] = {
                            "mean": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals)), 4),
                        }
                summary_entry["ard_lengthscales_summary"] = mean_ls

            all_results.setdefault("summary", {}).setdefault(split_type, {})[kernel_type] = summary_entry

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split_type in ["scaffold_split", "study_split"]:
        print(f"\n{split_type}:")
        for kernel_type in kernel_types:
            s = all_results["summary"][split_type][kernel_type]
            line = f"  {kernel_type}: R^2 = {s['r2_mean']:.4f} +/- {s['r2_std']:.4f}"
            if "ard_lengthscales_summary" in s:
                ls_str = ", ".join(
                    f"{col.replace('_molratio', '')}={v['mean']:.2f}"
                    for col, v in s["ard_lengthscales_summary"].items()
                )
                line += f"  [ARD: {ls_str}]"
            print(line)

    out_path = Path("models") / "product_kernel_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
