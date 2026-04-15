"""Audit sub-study heterogeneity and produce study definitions.

Outputs:
    stratification_report.json  -- machine-readable audit
    stratification_report.md    -- human-readable summary
    studies.json                -- study definitions for downstream experiments
"""

import json
from pathlib import Path

import pandas as pd

from LNPBO.data.lnpdb_bridge import load_lnpdb_full
from LNPBO.runtime_paths import package_root_from, workspace_root

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)
OUT_DIR = workspace_root(_PACKAGE_ROOT) / "experiments" / "data_integrity"
MIN_STUDY_SIZE = 200
MIN_SUBGROUP_SIZE = 100  # minimum rows in a (PMID, Model_target) subgroup
SEED_FRACTION = 0.25
BATCH_SIZE = 12
MAX_ROUNDS = 15
MIN_SEED = 30

CONTEXT_COLS = [
    "Model",
    "Model_type",
    "Model_target",
    "Experiment_method",
    "Experiment_ID",
    "Cargo",
    "Cargo_type",
    "Route_of_administration",
]


def audit_pmid_heterogeneity(df: pd.DataFrame) -> list[dict]:
    """For each PMID with >= MIN_STUDY_SIZE rows, report sub-study heterogeneity."""
    study_counts = df.groupby("Publication_PMID").size()
    qualifying = study_counts[study_counts >= MIN_STUDY_SIZE].index

    reports = []
    for pmid in qualifying:
        sub = df[df["Publication_PMID"] == pmid]
        n = len(sub)

        heterogeneity = {}
        for col in CONTEXT_COLS:
            if col not in sub.columns:
                continue
            vc = sub[col].fillna("_NA_").value_counts()
            heterogeneity[col] = {
                "n_unique": int(vc.shape[0]),
                "values": {str(k): int(v) for k, v in vc.items()},
            }

        # Per-Model_target z-score stats
        target_col = "Model_target" if "Model_target" in sub.columns else None
        target_stats = {}
        if target_col:
            for tgt, grp in sub.groupby(target_col):
                vals = grp["Experiment_value"].dropna()
                target_stats[str(tgt)] = {
                    "n": len(vals),
                    "mean": float(vals.mean()) if len(vals) > 0 else None,
                    "std": float(vals.std()) if len(vals) > 1 else None,
                    "min": float(vals.min()) if len(vals) > 0 else None,
                    "max": float(vals.max()) if len(vals) > 0 else None,
                    "frac": round(len(vals) / n, 3),
                }

        # Flag issues
        issues = []
        if target_col and heterogeneity.get("Model_target", {}).get("n_unique", 1) > 1:
            # Check if z-score distributions differ meaningfully between targets
            targets = sub.groupby("Model_target")["Experiment_value"]
            target_means = targets.mean()
            target_stds = targets.std()
            if target_means.std() > 0.5:
                issues.append(f"Model_target z-score means vary: std={target_means.std():.2f}")
            if target_stds.max() > 5:
                issues.append(f"Model_target z-score stds extreme: max={target_stds.max():.2f}")
            issues.append(f"{heterogeneity['Model_target']['n_unique']} unique Model_target values")

        if "Experiment_method" in heterogeneity and heterogeneity["Experiment_method"]["n_unique"] > 1:
            issues.append(f"{heterogeneity['Experiment_method']['n_unique']} unique Experiment_method values")

        reports.append(
            {
                "pmid": str(int(float(pmid))),
                "n_formulations": int(n),
                "n_unique_il": int(sub["IL_SMILES"].nunique()),
                "heterogeneity": heterogeneity,
                "target_zstats": target_stats,
                "issues": issues,
                "is_clean": len(issues) == 0,
            }
        )

    reports.sort(key=lambda r: -r["n_formulations"])
    return reports


def propose_studies(df: pd.DataFrame, reports: list[dict]) -> list[dict]:
    """Propose corrected study definitions, splitting multi-cell-line PMIDs.

    Primary split: by Model_type (cell line), since LNPDB z-scores are
    computed per (Experiment_ID, Model_type).  Each qualifying subgroup
    (n >= MIN_SUBGROUP_SIZE) becomes a separate study; smaller subgroups
    are excluded.

    For single-Model_type PMIDs with multiple Model_target values (e.g.
    different organs), the existing target-based splitting is preserved
    for biological relevance.
    """
    studies = []

    for rpt in reports:
        pmid_val = float(rpt["pmid"])
        sub = df[df["Publication_PMID"] == pmid_val]

        # Primary criterion: Model_type (cell line) heterogeneity
        model_types = rpt["heterogeneity"].get("Model_type", {})
        n_model_types = model_types.get("n_unique", 1)

        if n_model_types > 1:
            # Multi-cell-line PMID: split by Model_type
            mt_col = sub["Model_type"].fillna("_NA_")
            for mt_val in mt_col.unique():
                grp = sub[mt_col == mt_val]
                if len(grp) >= MIN_SUBGROUP_SIZE:
                    studies.append(
                        _make_study_def(
                            grp,
                            pmid_str=rpt["pmid"],
                            suffix=str(mt_val),
                            model_type_filter=str(mt_val),
                        )
                    )
            # No pooled version -- cross-cell-line pooling conflates z-scores
        else:
            # Single Model_type: check Model_target for biological splits
            targets = rpt["heterogeneity"].get("Model_target", {})
            n_targets = targets.get("n_unique", 1)

            if n_targets <= 1 or rpt["is_clean"]:
                studies.append(_make_study_def(sub, pmid_str=rpt["pmid"], suffix=None))
            else:
                target_vals = sub["Model_target"].fillna("_NA_")
                split_groups = []
                for tgt in target_vals.unique():
                    grp = sub[target_vals == tgt]
                    if len(grp) >= MIN_SUBGROUP_SIZE:
                        split_groups.append((tgt, grp))

                if len(split_groups) > 1:
                    for tgt, grp in split_groups:
                        suffix = str(tgt).replace(" ", "_").lower()
                        studies.append(
                            _make_study_def(grp, pmid_str=rpt["pmid"], suffix=suffix)
                        )
                else:
                    studies.append(
                        _make_study_def(sub, pmid_str=rpt["pmid"], suffix=None)
                    )

    studies.sort(key=lambda s: -s["n_formulations"])
    return studies


def _make_study_def(sub: pd.DataFrame, pmid_str: str, suffix: str | None,
                    model_type_filter: str | None = None) -> dict:
    n = len(sub)
    n_unique_il = int(sub["IL_SMILES"].nunique())
    n_unique_hl = int(sub["HL_name"].nunique()) if "HL_name" in sub.columns else 0
    n_unique_chl = int(sub["CHL_name"].nunique()) if "CHL_name" in sub.columns else 0
    n_unique_peg = int(sub["PEG_name"].nunique()) if "PEG_name" in sub.columns else 0

    il_ratio_std = float(sub["IL_molratio"].std()) if "IL_molratio" in sub.columns else 0.0

    if n_unique_il <= 1:
        study_type = "ratio_only"
    elif il_ratio_std > 1e-6:
        study_type = "il_diverse_variable_ratios"
    else:
        study_type = "il_diverse_fixed_ratios"

    n_seed = max(MIN_SEED, int(SEED_FRACTION * n))
    oracle_size = n - n_seed
    max_acquisitions = int(0.5 * oracle_size)
    n_rounds = min(MAX_ROUNDS, max(1, max_acquisitions // BATCH_SIZE))

    if n_unique_il <= 1:
        feature_type = "ratios_only"
    else:
        feature_type = "lantern_il_only"

    study_id = pmid_str if suffix is None else f"{pmid_str}_{suffix}"

    # Model_target summary
    model_targets = []
    if "Model_target" in sub.columns:
        model_targets = sub["Model_target"].fillna("_NA_").unique().tolist()

    return {
        "study_id": study_id,
        "pmid": pmid_str,
        "suffix": suffix,
        "model_type_filter": model_type_filter,
        "n_formulations": int(n),
        "n_unique_il": n_unique_il,
        "n_unique_hl": n_unique_hl,
        "n_unique_chl": n_unique_chl,
        "n_unique_peg": n_unique_peg,
        "il_ratio_std": il_ratio_std,
        "study_type": study_type,
        "feature_type": feature_type,
        "n_seed": n_seed,
        "n_rounds": n_rounds,
        "batch_size": BATCH_SIZE,
        "model_targets": model_targets,
        "is_pooled_mixed": False,
        "lnp_ids": sub["LNP_ID"].tolist() if "LNP_ID" in sub.columns else [],
    }


def write_markdown_report(reports: list[dict], corrected: list[dict], out_path: Path):
    lines = ["# Stratification Audit Report\n"]
    lines.append("**Date:** 2026-03-14\n")
    lines.append(f"**Total PMIDs audited:** {len(reports)}\n")

    clean = [r for r in reports if r["is_clean"]]
    mixed = [r for r in reports if not r["is_clean"]]
    lines.append(f"**Clean (single context):** {len(clean)}")
    lines.append(f"**Mixed (multiple contexts):** {len(mixed)}\n")

    # Mixed studies detail
    if mixed:
        lines.append("## Mixed-Context Studies\n")
        for r in sorted(mixed, key=lambda x: -x["n_formulations"]):
            lines.append(f"### PMID {r['pmid']} (N={r['n_formulations']})\n")
            lines.append("**Issues:**")
            for issue in r["issues"]:
                lines.append(f"- {issue}")
            lines.append("")

            if r["target_zstats"]:
                lines.append("| Model_target | N | Frac | Mean | Std | Min | Max |")
                lines.append("|---|---|---|---|---|---|---|")
                for tgt, st in sorted(r["target_zstats"].items(), key=lambda x: -x[1]["n"]):
                    lines.append(
                        f"| {tgt} | {st['n']} | {st['frac']:.1%} | "
                        f"{st['mean']:.3f} | {st['std']:.3f} | "
                        f"{st['min']:.2f} | {st['max']:.2f} |"
                    )
                lines.append("")

    # Corrected studies
    lines.append("## Corrected Study Definitions\n")
    non_pooled = [s for s in corrected if not s.get("is_pooled_mixed")]
    lines.append(f"**Total studies (after splitting):** {len(non_pooled)}")
    lines.append(f"**Original PMID count:** {len(reports)}\n")

    lines.append("| Study ID | PMID | Suffix | N | ILs | Type | Feature |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in non_pooled:
        lines.append(
            f"| {s['study_id']} | {s['pmid']} | {s['suffix'] or '-'} | "
            f"{s['n_formulations']} | {s['n_unique_il']} | "
            f"{s['study_type']} | {s['feature_type']} |"
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations, {df['Publication_PMID'].nunique()} studies")

    print("\nAuditing sub-study heterogeneity...")
    reports = audit_pmid_heterogeneity(df)

    print("\nProposing study definitions...")
    corrected = propose_studies(df, reports)

    # Write JSON reports
    json_path = OUT_DIR / "stratification_report.json"
    with open(json_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    corrected_path = OUT_DIR / "studies.json"
    # Don't write lnp_ids to the summary (too large); write separately
    corrected_slim = []
    for s in corrected:
        slim = {k: v for k, v in s.items() if k != "lnp_ids"}
        slim["n_lnp_ids"] = len(s.get("lnp_ids", []))
        corrected_slim.append(slim)
    with open(corrected_path, "w") as f:
        json.dump(corrected_slim, f, indent=2, default=str)
    print(f"Wrote {corrected_path}")

    # Write full corrected with LNP_IDs for benchmark use
    full_path = OUT_DIR / "studies_with_ids.json"
    with open(full_path, "w") as f:
        json.dump(corrected, f, indent=2, default=str)
    print(f"Wrote {full_path}")

    # Write markdown
    md_path = OUT_DIR / "stratification_report.md"
    write_markdown_report(reports, corrected, md_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    clean = [r for r in reports if r["is_clean"]]
    mixed = [r for r in reports if not r["is_clean"]]
    print(f"  Clean studies: {len(clean)}")
    print(f"  Mixed studies: {len(mixed)}")
    for r in mixed:
        targets = r["heterogeneity"].get("Model_target", {})
        print(
            f"    PMID {r['pmid']}: N={r['n_formulations']}, "
            f"{targets.get('n_unique', '?')} targets, issues: {'; '.join(r['issues'])}"
        )

    non_pooled = [s for s in corrected if not s.get("is_pooled_mixed")]
    print(f"\n  Corrected studies: {len(non_pooled)} (was {len(reports)} PMIDs)")
    for s in non_pooled:
        print(
            f"    {s['study_id']:>20s}  N={s['n_formulations']:>5}  ILs={s['n_unique_il']:>5}  type={s['study_type']}"
        )


if __name__ == "__main__":
    main()
