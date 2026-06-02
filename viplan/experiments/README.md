# Running experiments
1. Do this from the root directory of the repository.
2. Do only after installation is complete. Look at the [installation guide](../README.md) to install it.
3. Listed below are examples of how we ran our experiments.
4. They should all be executed in the mamba/conda environment
	```bash
	cd /home/navatm/Projects/ViPlanPO
	micromamba activate ./viplan_env
	```
3. We did not use the scripts in sh_scripts since some tasks were not relevant for us.
# Running iGibson server (simulator)

```bash
PORT=8000 make run
```
1. Use different `PORT` to run parallel experiments.
2. Apply the `PORT` here as the suffix in the below parameter for `--base_url` to connect the experiment to the simulator instance.
# Examples of running experiments
## ViLA Medium - all tasks
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/vila/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=DefaultVILAPolicy --use_predicate_groundings=False
```
### ViLA Medium - Food
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/vila/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=DefaultVILAPolicy --use_predicate_groundings=False --problem_filter_regex=".*?food.*?"
```
### ViLA Medium - Books
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8001 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/vila/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=DefaultVILAPolicy --use_predicate_groundings=False --problem_filter_regex=".*?books.*?"
```
### ViLA Medium - Groceries
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8002 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/vila/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=DefaultVILAPolicy --use_predicate_groundings=False --problem_filter_regex=".*?groceries.*?"
```
## ViLA Hard
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/hard --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/hard/vila/gpt-4.1 --max_steps 20 --seed 1 --policy_cls=DefaultVILAPolicy --use_predicate_groundings=False
```
## PolicyPlan Medium - all tasks
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8001 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/plan/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyPlan --use_predicate_groundings=False
```
### PolicyPlan Medium - Food
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/plan/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyPlan --use_predicate_groundings=False --problem_filter_regex=".*?food.*?"
```
### PolicyPlan Medium - Books
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8001 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/plan/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyPlan --use_predicate_groundings=False --problem_filter_regex=".*?books.*?"
```
### PolicyPlan Medium - Groceries
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8002 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/medium/plan/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyPlan --use_predicate_groundings=False --problem_filter_regex=".*?groceries.*?"
```
## PolicyPlan Hard
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8001 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/hard --prompt_path data/prompts/planning/vila_igibson_json.md --output_dir results/planning/igibson/hard/plan/gpt-4.1 --max_steps 20 --seed 1 --policy_cls=PolicyPlan --use_predicate_groundings=False
```
## CPP Hard Toys
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/hard --prompt_path data/prompts/benchmark/igibson/prompt_po.md --output_dir results/planning/igibson/hard/cpp/gpt-4.1 --max_steps 20 --seed 1 --policy_cls=PolicyCPP --use_predicate_groundings=False --problem_filter_regex=".*?toys.*?"
```
## CPP Hard Toys
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/hard --prompt_path data/prompts/benchmark/igibson/prompt_po.md --output_dir results/planning/igibson/hard/cpp/gpt-4.1 --max_steps 20 --seed 1 --policy_cls=PolicyCPP --use_predicate_groundings=False --problem_filter_regex=".*?toys.*?"
```
## CPP Medium food
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8000 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/benchmark/igibson/prompt_po.md --output_dir results/planning/igibson/medium/cpp/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyCPP --use_predicate_groundings=False --problem_filter_regex=".*?food.*?"
```
## CPP Medium books
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8001 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/benchmark/igibson/prompt_po.md --output_dir results/planning/igibson/medium/cpp/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyCPP --use_predicate_groundings=False --problem_filter_regex=".*?books.*?"
```
## CPP Medium groceries
```bash
PYTHONPATH=. python3 viplan/experiments/benchmark_igibson.py --base_url http://localhost:8002 --model_name gpt-4.1 --domain_file data/planning/igibson/domain.pddl --problems_dir data/planning/igibson/medium --prompt_path data/prompts/benchmark/igibson/prompt_po.md --output_dir results/planning/igibson/medium/cpp/gpt-4.1 --max_steps 15 --seed 1 --policy_cls=PolicyCPP --use_predicate_groundings=False --problem_filter_regex=".*?groceries.*?"
```
