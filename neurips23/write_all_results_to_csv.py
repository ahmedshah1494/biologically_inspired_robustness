import numpy as np
import pandas as pd
import os
import json
import random

from rblur.utils import load_json, aggregate_dicts

logdir = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/'

def aggregate_metrics(logdir, metrics_filename='metrics.json'):
    os.system(f"find {logdir} -maxdepth 2 -mindepth 2 -name {metrics_filename} > .find_tmp")
    with open('.find_tmp') as f:
        metrics_files = [l.strip() for l in f.readlines()]
    metrics = [load_json(p) for p in metrics_files]
    metrics = aggregate_dicts(metrics)
    return metrics

def load_metrics(logdir, metrics_filename='adv_metrics.json'):
    metrics = aggregate_metrics(logdir, metrics_filename).get('test_accs', {})
    return metrics

def create_all_metrics_df(logdir):
    os.system(f"find {logdir} -maxdepth 2 -mindepth 2 > logdirs.txt")

    with open('logdirs.txt') as f:
        logdirs = [l.strip() for l in f.readlines()]
    rows = []
    all_metrics = {os.path.basename(ld): load_metrics(ld) for ld in logdirs}
    for model_name, metrics in sorted(all_metrics.items(), key=lambda x: x[0]):
        r = {'model': model_name}
        for metric_name, values in metrics.items():
            r[metric_name] = np.mean(values)
            r[f'values_{metric_name}'] = len(values)
        rows.append(r)
    df = pd.DataFrame(rows)
    return df

df = create_all_metrics_df(logdir)
df.to_csv('ICLR22/all_metrics.csv')
# with open('ICLR22/all_metrics.json', 'w') as f:
#     json.dump(all_metrics, f)