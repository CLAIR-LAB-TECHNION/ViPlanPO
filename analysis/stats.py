import os

import pandas as pd

from analysis.consts import ROOT, splits
from analysis.source import load_results


def count_replans(results):
    rows = []
    for split in splits:
        for model, data in results.get(split, {}).items():
            problem_keys = [p for p in data.keys() if '.pddl' in str(p)]
            # Find how many questions were asked for the accuracy error
            prediction_count = 0
            for problem in problem_keys:
                rows.append({
                    'model': model,
                    'split': split,
                    'problem': problem,
                    'replans': len(data[problem]['replans'])
                })

    df = pd.DataFrame(rows)
    df['split'] = pd.Categorical(df['split'], categories=['simple', 'medium', 'hard'], ordered=True)
    df = df.sort_values(by=['model', 'split'])
    return df


if __name__ == '__main__':
    igibson_no_cot_experiment = os.path.join(ROOT, 'results', 'final', 'igibson', 'final_no_cot')
    igibson_no_cot_pred, igibson_no_cot_vila = load_results(igibson_no_cot_experiment)
    print(count_replans(igibson_no_cot_pred))
