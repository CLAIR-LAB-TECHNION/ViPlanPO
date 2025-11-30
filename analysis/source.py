import json
import os
from collections import defaultdict

from analysis.consts import splits


def load_results(experiment_folder):
    def _load(folder):
        assert os.path.exists(folder), f"{folder} does not exist"
        results = defaultdict(dict)
        for split in splits:
            results[split] = defaultdict(dict)
            split_folder = os.path.join(folder, split)
            if not os.path.exists(split_folder):
                print(f"Split folder {split_folder} does not exist")
                continue
            for model in os.listdir(split_folder):
                model_folder = os.path.join(split_folder, model)
                contents = os.listdir(model_folder)
                if not contents:
                    print(f"Model folder {model_folder} is empty")
                    continue
                if len(contents) > 1:
                    print(f"Model folder {model_folder} has more than one file, defaulting to most recent")
                latest = sorted(contents)[-1]
                with open(os.path.join(model_folder, latest)) as f:
                    results[split][model] = json.load(f)
        return results

    predicates_folder = os.path.join(experiment_folder, 'predicates')
    vila_folder       = os.path.join(experiment_folder, 'vila')

    pred_results = _load(predicates_folder)
    vila_results = _load(vila_folder)
    return pred_results, vila_results
