#!/bin/bash -l

# Default flags
RUN_CLOSED_SOURCE=false
RUN_PREDICATES=true
RUN_VILA=true

# Parse args into SBATCH (before --) and JOB (after --)
SBATCH_ARGS=()
JOB_ARGS=()
separator_seen=false
collect_sbatch=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_closed_source)
            RUN_CLOSED_SOURCE=true
            shift
            ;;
        --run_predicates)
            RUN_PREDICATES="$2"
            shift 2
            ;;
        --run_vila)
            RUN_VILA="$2"
            shift 2
            ;;
        --)
            separator_seen=true
            collect_sbatch=false
            shift
            ;;
        *)
            if [[ "$collect_sbatch" == "true" ]]; then
                SBATCH_ARGS+=("$1")
            else
                JOB_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Backward compatibility: if no `--` was provided, treat all non-wrapper args as JOB args
if [[ "$separator_seen" == "false" ]] && [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
    JOB_ARGS=("${SBATCH_ARGS[@]}")
    SBATCH_ARGS=()
fi

SCRIPT_DIR=$PWD/"sh_scripts/slurm_cluster/scripts"
echo "Script directory: $SCRIPT_DIR"

if [ ${#SBATCH_ARGS[@]} -gt 0 ]; then
    echo "sbatch options:"
    for a in "${SBATCH_ARGS[@]}"; do echo "  $a"; done
fi
if [ ${#JOB_ARGS[@]} -gt 0 ]; then
    echo "job script args:"
    for a in "${JOB_ARGS[@]}"; do echo "  $a"; done
fi

echo "Run closed source: $RUN_CLOSED_SOURCE"
echo "Run predicates:    $RUN_PREDICATES"
echo "Run vila:          $RUN_VILA"

# predicates (planning) benchmarks
if [[ "$RUN_PREDICATES" == "true" ]]; then
    if [[ "$RUN_CLOSED_SOURCE" == "true" ]]; then
        sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/benchmark_blocksworld_planning_array_cpu.sh" "${JOB_ARGS[@]}"
    fi
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/benchmark_blocksworld_planning_array_big.sh" "${JOB_ARGS[@]}"
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/benchmark_blocksworld_planning_array.sh"     "${JOB_ARGS[@]}"
fi

# vila benchmarks
if [[ "$RUN_VILA" == "true" ]]; then
    if [[ "$RUN_CLOSED_SOURCE" == "true" ]]; then
        sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/benchmark_blocksworld_vila_array_cpu.sh" "${JOB_ARGS[@]}"
    fi
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/benchmark_blocksworld_vila_array_big.sh" "${JOB_ARGS[@]}"
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/benchmark_blocksworld_vila_array.sh"     "${JOB_ARGS[@]}"
fi