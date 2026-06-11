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

# ---------------------------------------------------------------------------
# GPU memory hygiene — prevents the multi-day OOM/fragmentation stall where a
# big-CPU box launches more GPU-resident workers than one GPU can hold. All
# use ${VAR:-default} so an explicit env override still wins.
#   - expandable_segments: defeats CUDA caching-allocator fragmentation that
#     creeps up over long runs (the "fine for a day, then no progress" symptom)
#   - thread caps: each worker process otherwise spawns ncores BLAS threads;
#     N workers x ncores threads thrash the CPU once any worker falls back
#   - MIN_FREE_GIB: workers stay off CUDA unless this much VRAM is free
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export LNPBO_GPU_MIN_FREE_GIB="${LNPBO_GPU_MIN_FREE_GIB:-4}"

STUDIES_JSON="experiments/data_integrity/studies_with_ids.json"
N_WORKERS=4  # within-study default; capped to GPU capacity at runtime
TARGET="${1:-all}"

LOG_DIR="logs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== LNPBO Parallel Benchmark Runner ==="
echo "Target: $TARGET"
echo "Workers: $N_WORKERS"
echo "Logs: $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Max concurrent GPU-resident workers one GPU can hold, given a per-worker
# VRAM budget (LNPBO_GPU_GIB_PER_WORKER, default 4 GiB — a single GP fit can
# spike past 2 GiB). Returns a large sentinel (9999) when no GPU is present,
# so GPU does not constrain CPU-only fan-out.
#
# NOTE: bounds by a SINGLE GPU's total VRAM (the smallest, if several) rather
# than summing across GPUs, because get_device() in gp_bo.py always selects
# the default cuda device — every worker piles onto GPU 0. Reserves ~2 GiB
# for CUDA context + fragmentation headroom.
# ---------------------------------------------------------------------------
gpu_worker_cap() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 9999; return; }
    local total_mib
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | sort -n | head -1)
    [[ -z "$total_mib" || ! "$total_mib" =~ ^[0-9]+$ ]] && { echo 9999; return; }
    local total_gib=$(( total_mib / 1024 ))
    local per_worker="${LNPBO_GPU_GIB_PER_WORKER:-4}"
    [[ $per_worker -lt 1 ]] && per_worker=1
    local usable=$(( total_gib > 2 ? total_gib - 2 : total_gib ))
    local cap=$(( usable / per_worker ))
    [[ $cap -lt 1 ]] && cap=1
    echo "$cap"
}

# ---------------------------------------------------------------------------
# Within-study benchmark (load-balanced across workers)
# ---------------------------------------------------------------------------
run_within_study() {
    echo "--- Within-study benchmark ---"

    # Cap worker fan-out to what one GPU can hold (no-op when no GPU present).
    local gpu_cap
    gpu_cap=$(gpu_worker_cap)
    if [[ $gpu_cap -lt $N_WORKERS ]]; then
        echo "  GPU capacity caps within-study workers: $N_WORKERS -> $gpu_cap"
        N_WORKERS=$gpu_cap
    fi

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
# Resource detection — picks worker count from available CPUs and RAM.
# Overrideable via ABLATION_WORKERS env var. GPU presence is reported but
# does not constrain CPU-side fan-out (GP strategies handle device selection
# internally; concurrent processes share the GPU via CUDA's MPS/scheduler).
# ---------------------------------------------------------------------------
detect_ablation_workers() {
    local cpus mem_gb gpus
    if command -v nproc >/dev/null 2>&1; then
        cpus=$(nproc)
    elif [[ "$(uname)" == "Darwin" ]]; then
        cpus=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        cpus=4
    fi
    if [[ -r /proc/meminfo ]]; then
        mem_gb=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
    elif [[ "$(uname)" == "Darwin" ]]; then
        mem_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo 17179869184) / 1024 / 1024 / 1024))
    else
        mem_gb=16
    fi
    gpus=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    fi
    local by_gpu
    by_gpu=$(gpu_worker_cap)
    local n="${ABLATION_WORKERS:-}"
    if [[ -z "$n" ]]; then
        # 1 worker per 2 CPUs (reserving 2 for system) and 1 per 3GB RAM, then
        # bounded by GPU capacity (workers share one GPU), capped at 16.
        local by_cpu=$(( (cpus - 2) / 2 ))
        local by_mem=$(( mem_gb / 3 ))
        [[ $by_cpu -lt 1 ]] && by_cpu=1
        [[ $by_mem -lt 1 ]] && by_mem=1
        n=$by_cpu
        [[ $by_mem -lt $n ]] && n=$by_mem
        [[ $by_gpu -lt $n ]] && n=$by_gpu
        [[ $n -gt 16 ]] && n=16
    fi
    echo "$n|$cpus|$mem_gb|$gpus|$by_gpu"
}

# Extract per-condition labels from an ablation config so we can fan out
# across them via `run_ablation --condition <label>`. Returns one label
# per line; empty output means the ablation runs as a single process.
extract_condition_labels() {
    local cfg="$1"
    uv run python - "$cfg" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
labels = []
if "conditions" in cfg:
    labels = [c.get("label") for c in cfg["conditions"] if c.get("label")]
elif "configs" in cfg:
    labels = [c["name"] for c in cfg["configs"]]
elif "n_pcs_values" in cfg:
    labels = [f"pca{n}" for n in cfg["n_pcs_values"]]
elif "warmup_configs" in cfg:
    labels = [
        f"w{wc['warmup_size']}_{wc.get('selection', 'random')}_b{wc.get('bo_batch', 12)}"
        for wc in cfg["warmup_configs"]
    ]
elif "batch_sizes" in cfg:
    for bs in cfg["batch_sizes"]:
        for variant in cfg.get("variants", {}):
            labels.append(f"batch{bs}_{variant}")
for l in labels:
    print(l)
PY
}

# ---------------------------------------------------------------------------
# Ablations — fan out across conditions, scaled by detected resources.
# Ablations with no labeled conditions (kernel, small_study) run as one
# process; the rest split N ways and run their conditions in parallel.
# ---------------------------------------------------------------------------
run_ablations() {
    echo "--- Ablation experiments ---"

    local meta workers cpus mem_gb gpus
    meta=$(detect_ablation_workers)
    workers="${meta%%|*}"
    cpus=$(echo "$meta" | cut -d'|' -f2)
    mem_gb=$(echo "$meta" | cut -d'|' -f3)
    gpus=$(echo "$meta" | cut -d'|' -f4)
    local gpu_cap
    gpu_cap=$(echo "$meta" | cut -d'|' -f5)
    local gpu_note=""
    [[ "$gpus" -gt 0 ]] && gpu_note=" (GPU holds ~$gpu_cap)"
    echo "  Resources: $cpus CPUs, ${mem_gb}GB RAM, $gpus GPU(s)${gpu_note} → $workers worker(s) per ablation"
    echo "  (override with ABLATION_WORKERS=N or LNPBO_GPU_GIB_PER_WORKER=N)"

    local configs=(
        experiments/ablations/encoding/config.json
        experiments/ablations/batch_size/config.json
        experiments/ablations/pca/config.json
        experiments/ablations/budget/config.json
        experiments/ablations/warmup/config.json
        experiments/ablations/kappa/config.json
        experiments/ablations/kernel/config.json
        experiments/ablations/small_study/config.json
    )

    for cfg in "${configs[@]}"; do
        local name
        name=$(basename "$(dirname "$cfg")")

        local labels=()
        while IFS= read -r line; do
            [[ -n "$line" ]] && labels+=("$line")
        done < <(extract_condition_labels "$cfg" 2>/dev/null)

        local n_cond=${#labels[@]}
        if [ "$n_cond" -le 1 ]; then
            echo "  Ablation: $name (serial, no labeled conditions)"
            uv run python -m LNPBO.experiments.run_ablation \
                --config "$cfg" --resume \
                > "$LOG_DIR/ablation_${name}.log" 2>&1 \
                || echo "  WARNING: ablation $name failed"
        else
            local par=$workers
            [[ $par -gt $n_cond ]] && par=$n_cond
            echo "  Ablation: $name ($n_cond conditions, $par parallel workers)"
            local pids=() running=0
            for label in "${labels[@]}"; do
                while [ $running -ge $par ]; do
                    wait -n 2>/dev/null || wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                    running=$((running - 1))
                done
                local cond_log="$LOG_DIR/ablation_${name}_${label}.log"
                uv run python -m LNPBO.experiments.run_ablation \
                    --config "$cfg" --condition "$label" --resume \
                    > "$cond_log" 2>&1 &
                pids+=($!)
                running=$((running + 1))
            done
            for pid in "${pids[@]}"; do
                wait "$pid" || echo "  WARNING: ablation $name condition pid=$pid failed"
            done
        fi
    done
    echo "  Ablations done."
}

# ---------------------------------------------------------------------------
# Baselines (parallel — independent of each other)
# ---------------------------------------------------------------------------
run_baselines() {
    echo "--- Baselines ---"
    local pids=()
    local agile_root="${AGILE_ROOT:-../AGILE}"
    local comet_root="${COMET_ROOT:-../COMET}"
    local comet_weights_dir="${COMET_WEIGHTS_DIR:-${comet_root}/experiments/weights/weights}"

    uv run python -m LNPBO.benchmarks.baselines.predict_and_rank --resume \
        > "$LOG_DIR/baseline_predict_rank.log" 2>&1 &
    pids+=($!)

    (
        uv run python -m LNPBO.benchmarks.baselines.comet_wrapper --export &&
        uv run python -m LNPBO.benchmarks.baselines.comet_wrapper --run \
            --comet-repo "$comet_root" \
            --weights-dir "$comet_weights_dir" \
            --resume &&
        uv run python -m LNPBO.benchmarks.baselines.comet_wrapper --import-results
    ) > "$LOG_DIR/baseline_comet.log" 2>&1 &
    pids+=($!)

    AGILE_ROOT="$agile_root" uv run python -m LNPBO.benchmarks.baselines.agile_predictor --resume \
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
