import numpy as np
import pandas as pd
import os
import json
import random

from adversarialML.biologically_inspired_models.src.utils import load_json, aggregate_dicts
import numpy as np, scipy.stats as st

logdir = '/share/workhorse3/mshah1/biologically_inspired_models/icassp_logs/'

def dataframe_or(df, key, values):
    df_ = (df[key] == values[0])
    for v in values[1:]:
        df_ = df_ | (df[key] == v)
    return df_

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
            r[f'95CI_{metric_name}'] = st.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=st.sem(values))[1] - np.mean(values)
            r[f'values_{metric_name}'] = len(values)
        rows.append(r)
    df = pd.DataFrame(rows)
    return df

df = create_all_metrics_df(logdir)
df.to_csv('ICASSP23/all_metrics.csv')

table1_models = [
    'MNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'MNISTMLP2048U4L02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTMLP2048U4L02DropoutClassifier',
    'SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'SpeechCommandsMLP2048U4L02DropoutClassifier',
]
df[dataframe_or(df, 'model', table1_models)].to_csv('ICASSP23/table1.csv')

table2_models = [
    'MNISTAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'MNISTAdvTrainMLP2048U4L02DropoutClassifier',
    'FMNISTAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTAdvTrainMLP2048U4L02DropoutClassifier',
    'SpeechCommandsAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'SpeechCommandsAdvTrainMLP2048U4L02DropoutClassifier',
]
df[dataframe_or(df, 'model', table2_models)].to_csv('ICASSP23/table2.csv')

# table3_models = [
#     # 'MNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
#     # 'MNISTMLP2048U4L02DropoutClassifier',
#     'FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
#     'FMNISTMLP2048U4L02DropoutClassifier',
#     'SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
#     'SpeechCommandsMLP2048U4L02DropoutClassifier',
# ]
# keys = ['model'] + [c for c in df.columns if any([k in c for k in ['UNoise', 'RandOcc']])]
# df[dataframe_or(df, 'model', table1_models)][keys].to_csv('ICASSP23/table3.csv')

fig1_models = [
    'FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L8S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L4S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L2S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L1S0ClsS025ActOptLR02DropoutClassifier',
]
keys = ['model'] + [c for c in df.columns if any([k in c for k in ['APGD-0.0','APGD-0.05']])]
df[dataframe_or(df, 'model', fig1_models)].to_csv('ICASSP23/fig1.csv')