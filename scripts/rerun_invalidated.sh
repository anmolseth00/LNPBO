#!/usr/bin/env bash
# Re-runs the strategies invalidated by the Apr 15 fixes:
#   1b28203  CASMOPOLITAN (Wan et al. alignment)
#   9212886  Exact online conformal benchmark path
#   a1a6842  Manual Laplace evidence objective (affects RKB)
#   5e16839  GP batch acquisition semantics (ThompsonSamplingBatch + mixed-KB)
#   0fbf7dc  Batched random-forest kernel evaluation  (opt-in via INCLUDE_RF_KERNEL=1)
#   f553647  Exact FSBO path                          (runs by default; INCLUDE_FSBO=0 to skip)
#
# Usage (from the repo root, once `uv sync` has been run):
#     bash scripts/rerun_invalidated.sh
#     INCLUDE_RF_KERNEL=1 bash scripts/rerun_invalidated.sh   # also reruns RF-kernel strategies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi

RUN_TAG="${RUN_TAG:-rerun_$(date +%F)}"
OUT_ROOT="${OUT_ROOT:-benchmark_results/${RUN_TAG}}"
SEEDS="${SEEDS:-42,123,456,789,2024}"
WITHIN_STUDIES_JSON="${WITHIN_STUDIES_JSON:-experiments/data_integrity/studies.json}"
ABL_STUDIES_JSON="${ABL_STUDIES_JSON:-experiments/data_integrity/studies_with_ids.json}"
INCLUDE_RF_KERNEL="${INCLUDE_RF_KERNEL:-0}"
INCLUDE_FSBO="${INCLUDE_FSBO:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Strategies affected by commits 1b28203, 9212886, a1a6842, 5e16839.
# The 5e16839 blast radius covers every TS-batch strategy (ThompsonSamplingBatch
# was rewritten) and the mixed-KB path in select_batch_mixed.
MAIN_STRATEGIES="casmopolitan_ucb,casmopolitan_ei,\
discrete_xgb_online_conformal,\
lnpbo_rkb_logei,\
lnpbo_mixed_logei,lnpbo_mixed_ts,\
lnpbo_compositional_ts,lnpbo_tanimoto_ts,lnpbo_aitchison_ts,lnpbo_dkl_ts"

run_shell() {
    local cmd="$1"
    echo "+ $cmd"
    if [[ "$DRY_RUN" != "1" ]]; then
        bash -lc "$cmd"
    fi
}

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" tmp/rerun_configs

echo "== Within-study rerun: confirmed invalidated strategies =="
run_shell "uv run python -m LNPBO.benchmarks.benchmark \
  --studies-json \"$WITHIN_STUDIES_JSON\" \
  --strategies $MAIN_STRATEGIES \
  --seeds \"$SEEDS\" \
  --results-dir-override \"$OUT_ROOT/within_study_main\" \
  --resume \
  --log-file auto"

if [[ "$INCLUDE_RF_KERNEL" == "1" ]]; then
    echo "== Within-study rerun: conservative RF-kernel bucket =="
    run_shell "uv run python -m LNPBO.benchmarks.benchmark \
      --studies-json \"$WITHIN_STUDIES_JSON\" \
      --strategies lnpbo_rf_kernel_ts,lnpbo_rf_kernel_logei \
      --seeds \"$SEEDS\" \
      --results-dir-override \"$OUT_ROOT/within_study_rf_kernel\" \
      --resume \
      --log-file auto"
fi

echo "== Build targeted ablation configs =="
INCLUDE_RF_KERNEL="$INCLUDE_RF_KERNEL" uv run python - <<'PY'
import json
import os
from pathlib import Path

include_rf = os.environ.get("INCLUDE_RF_KERNEL", "0") == "1"

keep_main = [
    "casmopolitan_ucb",
    "casmopolitan_ei",
    "discrete_xgb_online_conformal",
    "lnpbo_rkb_logei",
    "lnpbo_mixed_logei",
    "lnpbo_mixed_ts",
    "lnpbo_compositional_ts",
    "lnpbo_tanimoto_ts",
    "lnpbo_aitchison_ts",
    "lnpbo_dkl_ts",
]
keep_with_optional_rf = keep_main + (["lnpbo_rf_kernel_ts", "lnpbo_rf_kernel_logei"] if include_rf else [])

jobs = [
    (
        "experiments/ablations/encoding/config.json",
        "tmp/rerun_configs/encoding_primary_targeted.json",
        ["casmopolitan_ucb"],
    ),
    (
        "experiments/ablations/encoding/config_full.json",
        "tmp/rerun_configs/encoding_full_targeted.json",
        keep_with_optional_rf,
    ),
    (
        "experiments/ablations/kappa/config_optimal.json",
        "tmp/rerun_configs/kappa_optimal_targeted.json",
        keep_with_optional_rf,
    ),
]

for src_name, out_name, keep in jobs:
    src = Path(src_name)
    cfg = json.loads(src.read_text())
    cfg["strategies"] = [s for s in cfg["strategies"] if s in keep]
    out = Path(out_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Wrote {out} with strategies: {cfg['strategies']}")
PY

echo "== Ablation rerun: primary encoding (CASMOPOLITAN-UCB only) =="
run_shell "uv run python -m LNPBO.experiments.run_ablation \
  --config tmp/rerun_configs/encoding_primary_targeted.json \
  --studies-json \"$ABL_STUDIES_JSON\" \
  --results-dir \"$OUT_ROOT/ablations/encoding_primary\" \
  --resume \
  2>&1 | tee \"$OUT_ROOT/logs/encoding_primary.log\""

echo "== Ablation rerun: full encoding (affected strategies only) =="
run_shell "uv run python -m LNPBO.experiments.run_ablation \
  --config tmp/rerun_configs/encoding_full_targeted.json \
  --studies-json \"$ABL_STUDIES_JSON\" \
  --results-dir \"$OUT_ROOT/ablations/encoding_full\" \
  --resume \
  2>&1 | tee \"$OUT_ROOT/logs/encoding_full.log\""

echo "== Ablation rerun: kappa-optimal benchmark slice (affected strategies only) =="
run_shell "uv run python -m LNPBO.experiments.run_ablation \
  --config tmp/rerun_configs/kappa_optimal_targeted.json \
  --studies-json \"$ABL_STUDIES_JSON\" \
  --results-dir \"$OUT_ROOT/ablations/kappa_optimal\" \
  --resume \
  2>&1 | tee \"$OUT_ROOT/logs/kappa_optimal.log\""

if [[ "$INCLUDE_FSBO" == "1" ]]; then
    echo "== Exact FSBO rerun =="
    run_shell "uv run python -m LNPBO.models.experimental.fsbo \
      2>&1 | tee \"$OUT_ROOT/logs/fsbo.log\""
    if [[ "$DRY_RUN" != "1" ]]; then
        cp models/fsbo_results.json "$OUT_ROOT/fsbo_results.json"
    else
        echo "+ cp models/fsbo_results.json \"$OUT_ROOT/fsbo_results.json\""
    fi
fi

echo
echo "Done."
echo "Outputs: $OUT_ROOT"
echo "Note: legacy gp_sklearn Thompson-sampling runs are not included here because there is no single canonical repo CLI for them."
