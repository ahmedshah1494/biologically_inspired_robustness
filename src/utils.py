import json
import pickle
import numpy as np
import os

import torch

def aggregate_metrics(logdir, metrics_filename='metrics.json'):
    metric_dicts = []
    for root, dirs, files in os.walk(logdir):
        if metrics_filename in files:
            m = load_json(os.path.join(root, metrics_filename))
            metric_dicts.append(m)
    agg_metrics = aggregate_dicts(metric_dicts)
    return agg_metrics

def aggregate_dicts(inp):
    if isinstance(inp, list) and all([isinstance(x, dict) for x in inp]):
        agg_dict = {}
        for d in inp:
            for k,v in d.items():
                agg_dict.setdefault(k, []).append(v)
        for k,v in agg_dict.items():
            agg_dict[k] = aggregate_dicts(v)
        return agg_dict
    else:
        return inp

def merge_iterables_in_dict(D):
    for k,v in D.items():
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                if (v[0].dim() > 0):
                    v = torch.cat(v, 0)
                else:
                    v = torch.tensor(v)
            elif isinstance(v[0], np.ndarray):
                if len(v[0].shape) > 0:
                    v = np.concatenate(v, 0)
                else:
                    v = np.ndarray(v)
            elif isinstance(v[0], list):
                v_merged = []
                for v_ in v:
                    v_merged.extend(v_)
                v = v_merged
        if isinstance(v, dict):
            v = merge_iterables_in_dict(v)
        D[k] = v
    return D

def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return d

def load_pickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d

def lazy_load_json(path):
    return lambda : load_json(path)

def lazy_load_pickle(path):
    return lambda : load_pickle(path)
     
def write_json(d, path):
    with open(path, 'w') as f:
        json.dump(d, f)

def write_pickle(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)