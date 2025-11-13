# Scripts to run the benchmark

The ViPlan benchmark is designed to be run on SLURM clusters, and the scripts in this directory are tailored for that purpose. If you are using a different cluster manager, you may need to modify the scripts accordingly, or directly run the Python scripts in the `viplan/experiments` directory.

> [!IMPORTANT]
> All sh_scripts are designed to be run from the root directory of the repository. (e.g. `cd ViPlan && ./sh_scripts/slurm_cluster/run_blocksworld.sh`)

The "big" scripts are designed to run bigger VLMs that require two GPUs and the "cpu" scripts are designed to run API models that don't require GPUs (although a GPU is still requested for the renderer).

The two main entry points are `run_blocksworld.sh` and `run_igibson.sh` (located at `sh_scripts/slurm_cluster` to run on SLURM clusters; at `sh_scripts/local` to run locally), which are designed to run the Blocksworld and Household environments. The parameters are :
- `--experiment_name` argument can be passed to all scripts and specifies a specific name that will be used to save the results
- `--run_predicates` boolean to determine whether to run experiments on the VLM-as-grounder setting
- `--run_vila` boolean to determine whether to run experiments on the VLM-as-planner setting
- `--run_closed_source` bolean to determine whether to run experiments using close-source models

Check the individual scripts for more details on the arguments.

## Passing sbatch options
- Place Slurm `sbatch` options before a `--` separator. All arguments after `--` are forwarded to the internal job scripts unchanged.
- When no `--` is provided, all non-wrapper arguments are treated as job script arguments (backward compatible behavior).

Examples:

```bash
# Set partition and time on all submitted jobs, then pass job args
./sh_scripts/slurm_cluster/run_blocksworld.sh -p gpu --time=02:00:00 -- \
	--experiment_name my_blocksworld_run --models some_model

# Run only VILA jobs on CPU partition, set array for all jobs
./sh_scripts/slurm_cluster/run_igibson.sh --run_predicates false -p cpu --array=0-49%5 -- \
	--experiment_name igibson_vila_only

# Backwards compatible: treat all args as job script args
./sh_scripts/slurm_cluster/run_blocksworld.sh --experiment_name legacy_behavior
```

Notes:
- The `--` separator is part of the wrapper interface to split sbatch options from job args; the wrapper will forward args to `sbatch` correctly.
- Quoting is preserved for complex flags like `--export=ALL,VAR="value with spaces"`.
- `--array` and GPU flags (e.g., `--gres=gpu:2`) apply to every job submitted by the wrapper.


Back to [Main Documentation](../README.md).