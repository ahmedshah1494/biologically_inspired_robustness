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

def _load_logs(logdir, model_filename='model_ckp.json', metrics_filename='metrics.json', 
                state_hist_filename='state_dict_hist.json', data_filename='data_and_preds.pkl',
                args_filename='task.pkl'):
    metrics_path = os.path.join(logdir, metrics_filename)
    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
    else:
        metrics = aggregate_metrics(logdir)
    args_path = os.path.join(logdir, args_filename)
    if os.path.exists(args_path):
        args = load_pickle(args_path)
    else:
        args = None
    metrics['test_accs'] = {float(k):v for k,v in metrics['test_accs'].items()}
    lazy_model_ckps = []
    lazy_state_hists = []
    lazy_data_and_preds = []
    model_metrics = []
    model_paths = []
    for root, _, files in os.walk(logdir):
        model_files = [f for f in files if f.startswith('model') and f.endswith('.pt')]
        if len(model_files) > 0:
            model_fp = os.path.join(root, model_files[-1])
            lazy_model_ckps.append(lazy_load_json(model_fp))
        # if state_hist_filename in files:
        #     sh_fp = os.path.join(root, state_hist_filename)
        #     lazy_state_hists.append(lazy_load_json(sh_fp))
        if data_filename in files:
            data_fp = os.path.join(root, data_filename)
            lazy_data_and_preds.append(lazy_load_pickle(data_fp))
        if metrics_filename in files and (root != logdir):
            m_fp = os.path.join(root, metrics_filename)
            model_metrics.append(load_json(m_fp))
            model_paths.append(root)
        if args is None and (args_filename in files):
            args_path = os.path.join(root, args_filename)
            args = args = load_pickle(args_path)
    log_dict = {
        'metrics': metrics,
        'models': lazy_model_ckps,
        'model_metrics': model_metrics,
        'model_paths': model_paths,
        'data_and_preds': lazy_data_and_preds,
        'args': args
    }
    return log_dict

def load_logs(logdirs, labels):
    if len(logdirs) == 1:
        logdicts = _load_logs(logdirs[0])
    else:
        logdicts = {}
        while len(labels) < len(logdirs):
            labels.append(None)
        for logdir, label in zip(logdirs, labels):
            if (label is None) or (label == 'na'):
                model_name = os.path.basename(logdir)
                i = 0
                mn = model_name
                while mn in logdicts:
                    i += 1
                    mn = f'{model_name}_{i}'
                label = mn
            logdicts[label] = _load_logs(logdir)
    return logdicts