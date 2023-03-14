import json
import pickle
import numpy as np
import os
import socket

import torch

def gethostname():
    return socket.gethostname()

def aggregate_metrics(logdir, metrics_filename='metrics.json'):
    metric_dicts = []
    expt_dirs = [f'{logdir}/{d}' for d in os.listdir(logdir)]
    for root in expt_dirs:
        files = os.listdir(root)
    # for root, dirs, files in os.walk(logdir):
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

def get_eps_from_logdict_key(s):
    if s.replace('.','',1).isdigit():
        eps = float(s)
        atkname = ''
    else:
        atkname, eps = s.split('-', 1)
        eps = float(eps)
    return atkname, eps

def _load_logs(logdir, model_filename='model_ckp.json', metrics_filename='adv_metrics.json',
                state_hist_filename='state_dict_hist.json', data_filename='data_and_preds.pkl',
                adv_battery_data_filename='adv_data_and_preds.pkl', adv_metrics_filename='adv_metrics.json',
                adv_succ_filename='adv_succ.json', args_filename='task.pkl', 
                randomized_smoothing_metrics_filename='randomized_smoothing_metrics.json',
                randomized_smoothing_data_filename='randomized_smoothing_preds_and_radii.pkl'):
    dataset = os.path.dirname(logdir)
    print(dataset, logdir)
    metrics_path = os.path.join(logdir, metrics_filename)
    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
    else:
        metrics = aggregate_metrics(logdir, metrics_filename)
        if len(metrics) == 0:
            metrics = aggregate_metrics(logdir, adv_metrics_filename)
    args_path = os.path.join(logdir, args_filename)
    if os.path.exists(args_path):
        args = load_pickle(args_path)
    else:
        args = None
    # metrics['test_accs'] = {k:v for k,v in metrics['test_accs'].items()}
    lazy_model_ckps = []
    lazy_state_hists = []
    lazy_data_and_preds = []
    lazy_adv_data_and_preds = []
    model_metrics = []
    model_adv_metrics = []
    model_adv_succ = []
    model_paths = []
    rs_metrics =[]
    lazy_rs_preds_and_radii = []
    # for root, _, files in os.walk(logdir):
    expt_dirs = [f'{logdir}/{d}' for d in os.listdir(logdir)]
    for root in expt_dirs:
        files = os.listdir(root)
        if 'source' in root:
            continue
        # model_files = [f for f in files if f.startswith('model') and f.endswith('.pt')]
        model_file_path = f'{root}/checkpoints/model_checkpoint.pt'
        if os.path.exists(model_file_path):
            model_fp = os.path.join(root, model_file_path)
            lazy_model_ckps.append(lazy_load_json(model_fp))
        # if len(model_files) > 0:
        #     model_fp = os.path.join(root, model_files[-1])
        #     lazy_model_ckps.append(lazy_load_json(model_fp))
        # if state_hist_filename in files:
        #     sh_fp = os.path.join(root, state_hist_filename)
        #     lazy_state_hists.append(lazy_load_json(sh_fp))
        if data_filename in files:
            data_fp = os.path.join(root, data_filename)
            lazy_data_and_preds.append(lazy_load_pickle(data_fp))
        if adv_battery_data_filename in files:
            data_fp = os.path.join(root, adv_battery_data_filename)
            lazy_adv_data_and_preds.append(lazy_load_pickle(data_fp))
        if adv_metrics_filename in files:
            am_fp = os.path.join(root, adv_metrics_filename)
            model_adv_metrics.append(load_json(am_fp))
        if adv_succ_filename in files:
            as_fp = os.path.join(root, adv_succ_filename)
            model_adv_succ.append(load_json(as_fp))
        if randomized_smoothing_data_filename in files:
            data_fp = os.path.join(root, randomized_smoothing_data_filename)
            lazy_rs_preds_and_radii.append(lazy_load_pickle(data_fp))
        if randomized_smoothing_metrics_filename in files:
            am_fp = os.path.join(root, randomized_smoothing_metrics_filename)
            rs_metrics.append(load_json(am_fp))
        if metrics_filename in files and (root != logdir):
            m_fp = os.path.join(root, metrics_filename)
            model_metrics.append(load_json(m_fp))
            model_paths.append(root)
        # if args is None and (args_filename in files):
        #     args_path = os.path.join(root, args_filename)
        #     args = load_pickle(args_path)
    if len(model_metrics) == 0:
        model_metrics = model_adv_metrics
    log_dict = {
        'metrics': metrics,
        'models': lazy_model_ckps,
        'model_metrics': model_metrics,
        'model_paths': model_paths,
        'model_adv_metrics': model_adv_metrics,
        'model_adv_succ': model_adv_succ,
        'data_and_preds': lazy_data_and_preds,
        'adv_data_and_preds': lazy_adv_data_and_preds,
        'rs_metrics': rs_metrics,
        'rs_preds_and_radii': lazy_rs_preds_and_radii,
        'args': args,
        'dataset': dataset
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

def _compute_area_under_curve(x, y):
    x = np.array(x)
    y = np.array(y)
    sorted_idx = sorted(range(len(x)), key=lambda i: x[i])
    x = x[sorted_idx]
    y = y[sorted_idx]
    total_area = 0
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        assert h >= 0
        a = y[i]
        b = y[i-1]
        area = h*(a+b)/2
        total_area += area
    return total_area

def get_model_checkpoint_paths(d):
    model_ckp_dirs = []
    for root, dirs, files in os.walk(d):
        model_files = [f for f in files if f.startswith('model') and f.endswith('.pt')]
        if len(model_files) > 0:
            model_file = model_files[0]
            model_ckp_dirs.append(os.path.join(root, model_file))
    return model_ckp_dirs

def recursive_dict_update(dsrc, dtgt):
    for ksrc, vsrc in dsrc.items():
        if ksrc in dtgt:
            vtgt = dtgt[ksrc]
            if isinstance(vtgt, dict) and isinstance(vsrc, dict):
                recursive_dict_update(vsrc, vtgt)
            else:
                dtgt[ksrc] = vsrc
        else:
            dtgt[ksrc] = vsrc

def variable_length_sequence_collate_fn(data):
    wavs, txts = zip(*data)
    wavs = [w.transpose(0,1) for w in wavs]
    txts = [torch.LongTensor(t) for t in txts]
    input_lengths = torch.LongTensor([len(w) for w in wavs])
    target_lengths = torch.LongTensor([len(t) for t in txts])
    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0)
    txts = torch.nn.utils.rnn.pad_sequence(txts, batch_first=True, padding_value=2)
    return wavs, txts, input_lengths, target_lengths