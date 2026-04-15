#!/usr/bin/env bash
# Parallel benchmark runner — splits studies across N workers.
#
# Usage:
#   ./scripts/run_parallel.sh                    # all sections
#   ./scripts/run_parallel.sh within-study       # within-study only
#   ./scripts/run_parallel.sh ablations          # ablations only
#   ./scripts/run_parallel.sh baselines          # baselines
#   ./scripts/run_parallel.sh sensitivity        # sensitivity analyses
#   ./scripts/run_parallel.sh analysis           # cross-study + calibration
#   ./scripts/run_parallel.sh figures            # figures only
#
# Each within-study worker writes to its own study subdirectory — no conflicts.

set -euo pipefail
cd "$(dirname "$0")/.."

STUDIES_JSON="experiments/data_integrity/studies_with_ids.json"
N_WORKERS=4  # 4 workers × ~3 cores each = 12 cores
TARGET="${1:-all}"

LOG_DIR="logs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== LNPBO Parallel Benchmark Runner ==="
echo "Target: $TARGET"
echo "Workers: $N_WORKERS"
echo "Logs: $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Within-study benchmark (load-balanced across workers)
# ---------------------------------------------------------------------------
run_within_study() {
    echo "--- Within-study benchmark ---"

    # Load-balance studies into N_WORKERS groups by total formulations.
    # Distribute by study_id (not PMID) to avoid conflicts on sub-studies.
    local groups
    groups=$(uv run python -c "
import json, sys

studies = json.load(open('$STUDIES_JSON'))
# Filter out pooled_mixed
studies = [s for s in studies if not s.get('is_pooled_mixed')]
# Sort by size descending
studies.sort(key=lambda s: -s['n_formulations'])

N = $N_WORKERS
buckets = [[] for _ in range(N)]
loads = [0] * N

# Greedy load balancing: assign each study to the lightest bucket
for s in studies:
    i = loads.index(min(loads))
    buckets[i].append(s['study_id'])
    loads[i] += s['n_formulations']

for i, (bucket, load) in enumerate(zip(buckets, loads)):
    # Pass study_ids via --pmids (the flag matches both PMIDs and study_ids)
    study_ids = ','.join(bucket)
    print(f'{i}|{study_ids}|{load}')
")

    local pids=()
    while IFS='|' read -r idx study_ids load; do
        echo "  Worker $idx: studies=$study_ids (N≈$load formulations)"
        uv run python -m LNPBO.benchmarks.benchmark \
            --studies-json "$STUDIES_JSON" \
            --pmids "$study_ids" \
            --resume \
            > "$LOG_DIR/within_study_worker${idx}.log" 2>&1 &
        pids+=($!)
    done <<< "$groups"

    echo "  Launched ${#pids[@]} workers. Waiting..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: Worker PID $pid exited with error"
            failed=$((failed + 1))
        fi
    done

    if [ $failed -gt 0 ]; then
        echo "  $failed worker(s) had errors. Check logs in $LOG_DIR/"
    else
        echo "  All within-study workers completed."
    fi

    # Aggregate (single process)
    echo "  Running aggregation..."
    uv run python -m LNPBO.benchmarks.benchmark \
        --studies-json "$STUDIES_JSON" \
        --aggregate-only \
        > "$LOG_DIR/within_study_aggregate.log" 2>&1
    echo "  Aggregation done."
}

# ---------------------------------------------------------------------------
# Ablations (sequential — each config is independent)
# ---------------------------------------------------------------------------
run_ablations() {
    echo "--- Ablation experiments ---"
    local configs=(
        experiments/ablations/encoding/config_full.json
        experiments/ablations/batch_size/config.json
        experiments/ablations/pca/config.json
        experiments/ablations/budget/config.json
        experiments/ablations/warmup/config.json
        experiments/ablations/kappa/config.json
        experiments/ablations/kernel/config.json
        experiments/ablations/small_study/config.json
    )

    # Run ablations 2 at a time (each uses ~4 cores)
    local i=0
    while [ $i -lt ${#configs[@]} ]; do
        local pids=()
        for j in 0 1; do
            local idx=$((i + j))
            if [ $idx -lt ${#configs[@]} ]; then
                local cfg="${configs[$idx]}"
                local name=$(basename "$(dirname "$cfg")")
                echo "  Launching ablation: $name"
                uv run python -m LNPBO.experiments.run_ablation \
                    --config "$cfg" --resume \
                    > "$LOG_DIR/ablation_${name}.log" 2>&1 &
                pids+=($!)
            fi
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "  WARNING: ablation PID $pid failed"
        done
        i=$((i + 2))
    done
    echo "  Ablations done."
}

# ---------------------------------------------------------------------------
# Baselines (parallel — independent of each other)
# ---------------------------------------------------------------------------
run_baselines() {
    echo "--- Baselines ---"
    local pids=()

    uv run python -m LNPBO.benchmarks.baselines.predict_and_rank \
        > "$LOG_DIR/baseline_predict_rank.log" 2>&1 &
    pids+=($!)

    uv run python -m LNPBO.benchmarks.baselines.comet_infer \
        > "$LOG_DIR/baseline_comet.log" 2>&1 &
    pids+=($!)

    uv run python -m LNPBO.benchmarks.baselines.agile_predictor \
        > "$LOG_DIR/baseline_agile.log" 2>&1 &
    pids+=($!)

    for pid in "${pids[@]}"; do
        wait "$pid" || echo "  WARNING: baseline PID $pid failed"
    done
    echo "  Baselines done."
}

# ---------------------------------------------------------------------------
# Sensitivity analyses (parallel)
# ---------------------------------------------------------------------------
run_sensitivity() {
    echo "--- Sensitivity analyses ---"
    local pids=()

    uv run python -m LNPBO.benchmarks.threshold_sensitivity \
        > "$LOG_DIR/sensitivity_threshold.log" 2>&1 &
    pids+=($!)

    uv run python -m LNPBO.benchmarks.substudy_sensitivity \
        > "$LOG_DIR/sensitivity_substudy.log" 2>&1 &
    pids+=($!)

    uv run python -m LNPBO.benchmarks.noise_sensitivity \
        > "$LOG_DIR/sensitivity_noise.log" 2>&1 &
    pids+=($!)

    for pid in "${pids[@]}"; do
        wait "$pid" || echo "  WARNING: sensitivity PID $pid failed"
    done

    # hyperparam sensitivity is heavier, run alone
    uv run python -m LNPBO.benchmarks.hyperparam_sensitivity --resume \
        > "$LOG_DIR/sensitivity_hyperparam.log" 2>&1
    echo "  Sensitivity analyses done."
}

# ---------------------------------------------------------------------------
# Cross-study transfer + calibration
# ---------------------------------------------------------------------------
run_analysis() {
    echo "--- Analysis ---"
    local pids=()

    uv run python -m LNPBO.experiments.cross_study_transfer \
        > "$LOG_DIR/analysis_cross_study.log" 2>&1 &
    pids+=($!)

    uv run python -m LNPBO.experiments.calibration_analysis \
        > "$LOG_DIR/analysis_calibration.log" 2>&1 &
    pids+=($!)

    for pid in "${pids[@]}"; do
        wait "$pid" || echo "  WARNING: analysis PID $pid failed"
    done
    echo "  Analysis done."
}

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
run_figures() {
    echo "--- Figures ---"
    uv run python -m LNPBO.benchmarks.analyze_within_study --aggregate-only \
        > "$LOG_DIR/figures_aggregate.log" 2>&1
    cd paper && uv run python make_all_figures.py \
        > "../$LOG_DIR/figures_generate.log" 2>&1
    echo "  Figures done."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$TARGET" in
    within-study)   run_within_study ;;
    ablations)      run_ablations ;;
    baselines)      run_baselines ;;
    sensitivity)    run_sensitivity ;;
    analysis)       run_analysis ;;
    figures)        run_figures ;;
    all)
        run_within_study
        run_ablations
        run_baselines
        run_sensitivity
        run_analysis
        run_figures
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Valid: within-study, ablations, baselines, sensitivity, analysis, figures, all"
        exit 1
        ;;
esac

echo ""
echo "=== Done: $TARGET ==="
echo "Logs: $LOG_DIR/"
