from argparse import ArgumentParser
import argparse
from collections import OrderedDict
from itertools import combinations
import itertools
import json
import os
import pickle
import re
import time
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE, MDS
from scipy.special import softmax
from scipy.stats.mstats import f_oneway
from matplotlib.animation import FuncAnimation
import torch
from tqdm import tqdm
from adversarialML.biologically_inspired_models.src.utils import _compute_area_under_curve, load_logs, get_eps_from_logdict_key, load_json, aggregate_dicts, lazy_load_pickle
from adversarialML.biologically_inspired_models.src.ICLR22.write_all_results_to_csv import create_all_metrics_df
import re
# plt.rcParams['text.usetex'] = True

def dataframe_or(df, key, values):
    df_ = (df[key] == values[0])
    for v in values[1:]:
        df_ = df_ | (df[key] == v)
    return df_

def create_data_df(logdicts, plot_config):
    data = []
    for model_name, logdict in logdicts.items():
        print(model_name)
        test_acc = logdict.get('metrics', {}).get('test_accs', None)
        if test_acc is None:
            print(f'Data for {model_name} was not found. Skipping...')
            continue
        metrics_to_plot = plot_config[model_name][-1]
        print(model_name, metrics_to_plot)
        # best_model_idx = np.argmax(test_acc[min(test_acc.keys())])
        for atkstr, accs in test_acc.items():
            atkname, eps = get_eps_from_logdict_key(atkstr)
            if (('L2' in atkname) and (eps <= 2.5)) or (eps <= 0.016):
                if atkname in metrics_to_plot:
                    # accs = [accs[best_model_idx]]
                    for i,a in enumerate(accs):
                        # dp = data_and_preds[i]()
                        # try:
                        #     num_data_points = len(dp[atkstr]['Y'])
                        #     # print(i, model_name, num_data_points)
                        #     if num_data_points < 10000:
                        #         print(f'{model_name}-{atkstr}-{i} has {num_data_points} points but expected 10_000')
                        # except:
                        #     print(f'{model_name} may be missing some data. Expected {atkstr} to be in data_and_preds, but data_and_preds contains {list(dp.keys())}')
                        r = {
                            'Method': model_name,
                            f'Perturbation Distance ‖ϵ‖{2 if "L2" in atkname else "∞"}': float(eps),
                            'Accuracy': a*100,
                            'Attack': atkname
                        }
                        data.append(r)
    df = pd.DataFrame(data)
    return df

def create_many_fixation_data_df(logdicts, plot_config):
    data = []
    for model_name, logdict in logdicts.items():
        print(model_name)
        test_acc = logdict.get('many_fixation_metrics', {})
        if test_acc is None:
            print(f'Data for {model_name} was not found. Skipping...')
            continue
        # best_model_idx = np.argmax(test_acc[min(test_acc.keys())])
        for atkstr, accs in test_acc.items():
            print(atkstr)
            epsstr, nstr = atkstr.split('_')[1:]
            eps = float(epsstr.split('=')[-1])
            npoints = int(nstr.split('=')[-1])
            # accs = [accs[best_model_idx]]
            for i,a in enumerate(accs):
                r = {
                    'Method': model_name,
                    f'Perturbation Distance ‖ϵ‖∞': float(eps),
                    'Accuracy': a*100,
                    'Number of Fixation Points': npoints
                }
                data.append(r)
    df = pd.DataFrame(data)
    return df

def plot_training_method_comparison(df, outdir, plot_config):
    hue_order = plot_config
    plt.figure(figsize=(30,10))
    sns.set_style("whitegrid")
    plt.ylim(0, 1.)
    sns.barplot(x='test_eps', y='acc', hue='model_name', hue_order=hue_order, data=df)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar.png'))

def get_outdir(model_log_dir, outdir, final_dirname):
    dirs = np.array([x.split('/')[-3:] for x in model_log_dir])
    unique_dirs = [np.unique(dirs[:,i]) for i in range(dirs.shape[1])]
    concat_dirs = lambda a: '+'.join(a)
    outdir = [outdir] + [concat_dirs(d) for d in unique_dirs]
    if final_dirname is not None:
        outdir[-1] = final_dirname
    outdir = os.path.join(*outdir)
    return outdir

def get_logdict(plot_config):
    logdirs_and_labels = {(ld, label) for label, (ld, _) in plot_config.items()}
    logdirs, labels = zip(*logdirs_and_labels)
    logdict = {}
    for logdir, label in logdirs_and_labels:
        expdirs = [os.path.join(logdir, x) for x in os.listdir(logdir)]
        metric_files = [os.path.join(d, 'adv_metrics.json') for d in expdirs]
        many_fixation_metric_files = [os.path.join(d, 'many_fixations_results.json') for d in expdirs]
        many_fixation_metrics = aggregate_dicts([load_json(x) for x in many_fixation_metric_files if os.path.exists(x)])
        metrics = aggregate_dicts([load_json(x) for x in metric_files if os.path.exists(x)])
        rs_files = [os.path.join(d, 'randomized_smoothing_preds_and_radii.pkl') for d in expdirs]
        lazy_rs_preds_and_radii = [lazy_load_pickle(x) for x in rs_files if os.path.exists(x)]
        logdict[label] = {'metrics': metrics, 'rs_preds_and_radii': lazy_rs_preds_and_radii, 'many_fixation_metrics':many_fixation_metrics}
    return logdict

def load_cc_results(plot_config, path_and_label_file):
    def compute_accuracy_per_severity(lnp_pth, cns_list):
        print(lnp_pth)
        lnp = np.loadtxt(lnp_pth, skiprows=1, delimiter=',')
        is_correct = (lnp[:,0] == lnp[:,1]).astype(float)

        cns2acc = {}
        for cns, ic in zip(cns_list, is_correct):
            cns2acc.setdefault(tuple(cns), []).append(ic)
        cns2acc = {k: np.mean(v) for k,v in cns2acc.items()}
        return cns2acc

        # corr_types = np.array([x[0] for x in cns_list])
        # sevs = np.array([int(x[1]) for x in cns_list])
        # sev2acc = {}
        # for sev in sorted(set(sevs)):
        #     sev2acc[sev] = is_correct[sevs == sev].mean()
        # print(sev2acc)
        # return sev2acc

    with open(path_and_label_file) as f:
        fnames = [l.split(',')[-1].split('/')[-1].split('.')[0] for l in f.readlines()]
        corruption_and_severity = [fn[:fn.index('-')+2].split('-') for fn in fnames]
        corruption_and_severity = [(x[0],int(x[1])) for x in corruption_and_severity]
    logdirs_and_labels = [(ld, label, atk[0]) for label, (ld, atk) in plot_config.items()]
    logdict = {}
    for logdir, label, atk in logdirs_and_labels:
        expdirs = [os.path.join(logdir, x) for x in os.listdir(logdir)]
        metric_files = [os.path.join(d, 'per_attack_results', f'{atk}-0.0_label_and_preds.csv') for d in expdirs]
        print(metric_files)
        metrics = aggregate_dicts([compute_accuracy_per_severity(x, corruption_and_severity) for x in metric_files if os.path.exists(x)])
        logdict[label] = {'metrics': metrics}

    # rows = []
    # for method, ld in logdict.items():
    #     for sev, accs in ld['metrics'].items():
    #         for a in accs:
    #             r = {
    #                 'Method': method,
    #                 'Corruption Severity': sev,
    #                 "Accuracy": a
    #             }
    #             rows.append(r)
    class2corr = {
        'noise': ['gaussian_noise',
                'shot_noise',
                'impulse_noise',
                'speckle_noise',],
        'blur': ['defocus_blur',
                'glass_blur',
                'motion_blur',
                'zoom_blur',
                'gaussian_blur',],
        'weather': ['snow',
                    'frost',
                    'fog',
                    'brightness',
                    'spatter',],
        'digital': ['contrast',
                    'elastic_transform',
                    'pixelate',
                    'jpeg_compression',
                    'saturate',]
    }
    corr2class = {}
    for cl,cors in class2corr.items():
        for c in cors:
            corr2class[c] = cl

    rows = []
    for method, ld in logdict.items():
        for (corr, sev), accs in ld['metrics'].items():
            for a in accs:
                r = {
                    'Method': method,
                    'Corruption Severity': sev,
                    'Corruption Method': corr,
                    'Corruption Type': corr2class[corr],
                    'Accuracy': a
                }
                rows.append(r)
    df = pd.DataFrame(rows)
    return df

def maybe_create_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d

log_root = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs'
outdir_root = 'ICLR22/visualizations'
sns.set(font_scale=1.5)

def plot_cifar10_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        # ('GBlur (σ=1.5)', (f'{log_root}/cifar10-0.0/Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('R-Warp', (f'{log_root}/cifar10-0.0/Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_cifar10_pgdl2_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGDL2'])),
        # ('GBlur (σ=1.5)', (f'{log_root}/cifar10-0.0/Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/cifar10-0.0/Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGDL2'])),
        # ('G-Noise', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseCyclicLRAutoAugmentWideResNet4x22', ['APGDL2'])),
        ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_ecoset10_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf-2.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset100_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianBlurCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset100_pgdinf_randaug_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Blur (RandAug)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur (no RandAug)', (f'{log_root}/ecoset100-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRXResNet2x18', ['5FixationAPGD'])),

    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_randaug.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset_pgdinf_results_with_affine():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('- w/ 5 Affine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['5RandAug'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] >= 0.002])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_with_affine.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset10_pgdl2_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        # ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_ecoset100_pgdl2_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianBlurCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        # ('G-Noise', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, estimator=max, ci=None, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_ecoset_pgdl2_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df['Perturbation Distance ‖ϵ‖2'] > 0.) & (df['Perturbation Distance ‖ϵ‖2'] != 1)])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_imagenet_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] >= 0.002])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
    plt.close()

def plot_imagenet_pgdl2_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df['Perturbation Distance ‖ϵ‖2'] != 1)])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_imagenet_pgd_results():
    plt.figure(figsize=(30,5))
    plt.subplot(1, 4, 1)
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*20 for i in range(5)], [i*20 for i in range(5)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()

    plt.subplot(1, 4, 2)
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    print(df)
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df['Perturbation Distance ‖ϵ‖2'] != 1)])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*20 for i in range(5)], [i*20 for i in range(5)])
    # plt.legend([],[], frameon=False)

    plt.subplot(1, 4, 3)
    plt.title('σ=0.125')
    plot_config = OrderedDict([
        # ('R-Warp-CFI', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('R-Blur-CFI (σ=0.125)', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', '0.25'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 200)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        # plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.lineplot(x='radius', y='accuracy', hue='model_name', hue_order=plot_config.keys(), data=df[df['σ'] == 0.125])

    plt.subplot(1, 4, 4)
    plt.title('σ=0.25')
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        # plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.lineplot(x='radius', y='accuracy', hue='model_name', hue_order=plot_config.keys(), data=df[df['σ'] == 0.25])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'pgd+rs.png'))
    plt.close()



def plot_cifar10_pgdinf_training_ablation_results():
    plot_config = OrderedDict([
        ('VDT-5FI', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('VDT-CFI', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['CenteredAPGD'])),
        ('FDT-5FI', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('NoNoise', (f'{log_root}/cifar10-0.0/Cifar10RetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('OnlyColor', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurOnlyColorWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('NoBlur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurNoBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('G-Noise', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('GBlur-WNoise', (f'{log_root}/cifar10-0.0/Cifar10NoisyGaussianBlurCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('GBlur-NoNoise', (f'{log_root}/cifar10-0.0/Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        # ('FDT-CFI', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    plot = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config,
                # data=df[(df['Perturbation Distance ‖ϵ‖∞'] == 0.) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.008)]
                data=df[dataframe_or(df, 'Perturbation Distance ‖ϵ‖∞', [0., .008])]
                )
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_ablation_acc_bar_linf.png'))
    plt.close()

# def plot_cifar10_pgdl2_training_ablation_results():
#     plot_config = OrderedDict([
#         ('VDT-5FI', (f'{log_root}/cifar10-0.0/Cifar10RetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGDL2'])),
#         ('VDT-CFI', (f'{log_root}/cifar10-0.0/Cifar10RetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['CenteredAPGDL2'])),
#         ('FDT-5FI', (f'{log_root}/cifar10-0.0/Cifar10RetinaBlurCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGDL2'])),
#         ('FDT-CFI', (f'{log_root}/cifar10-0.0/Cifar10RetinaBlurCyclicLRAutoAugmentWideResNet4x22', ['CenteredAPGDL2'])),
#     ])

#     logdicts = get_logdict(plot_config)
#     df = create_data_df(logdicts, plot_config)
#     outdir = maybe_create_dir(f'{outdir_root}/cifar10')
#     sns.set_style("whitegrid")
#     sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, 'training_ablation_acc_bar_l2.png'))
#     plt.close()

def plot_ecoset10_pgdinf_training_ablation_results_1():
    plot_config = OrderedDict([
        ('Everything', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('No 5Fixations', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        ('No VDT', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('No Noise', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('No Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500NoBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('Only Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('Only Desaturation', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500OnlyColorWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('Non-Adaptive-Blur with Noise', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        # ('Only Non-Adaptive-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        # ('Only Non-Adaptive Desaturation', (f'{log_root}/ecoset10-0.0/Ecoset10GreyscaleCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    df = df[dataframe_or(df, 'Perturbation Distance ‖ϵ‖∞', [0., .008])]
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    print(df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        g = sns.catplot(x='Method', y='Accuracy', hue='Is Perturbed', kind='bar', data=df, aspect=1.75, legend=False, order=plot_config)
        ax = g.facet_axis(0, 0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    plt.savefig(os.path.join(outdir, 'training_ablation_acc_bar_linf_1.png'))
    plt.close()

def plot_ecoset10_pgdinf_training_ablation_results_2():
    plot_config = OrderedDict([
        # ('Everything', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('No 5Fixations', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        # ('No VDT', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('No Noise', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('No Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500NoBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('Only Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('Only Desaturation', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500OnlyColorWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('Non-Adaptive-Blur with Noise', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('Only Non-Adaptive-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('Only Non-Adaptive Desaturation', (f'{log_root}/ecoset10-0.0/Ecoset10GreyscaleCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    df = df[dataframe_or(df, 'Perturbation Distance ‖ϵ‖∞', [0., .008])]
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    print(df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        g = sns.catplot(x='Method', y='Accuracy', hue='Is Perturbed', kind='bar', data=df, aspect=1.75, legend=False, order=plot_config)
        ax = g.facet_axis(0, 0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    plt.savefig(os.path.join(outdir, 'training_ablation_acc_bar_linf_2.png'))
    plt.close()

def create_rs_dataframe(logdicts, plot_config, max_points=10000):
    radius_step = 0.01
    data = []
    for model_name, logdict in logdicts.items():
        for model_data in logdict['rs_preds_and_radii']:
            metrics_to_plot = plot_config[model_name][-1]
            model_data = model_data()
            y = np.array(model_data['Y'])[:max_points]
            pnr_for_sigma = model_data['preds_and_radii']
            print(model_name, metrics_to_plot, pnr_for_sigma.keys())
            for sigma, pnr in pnr_for_sigma.items():
                if sigma not in metrics_to_plot:
                    continue
                if isinstance(sigma, str):
                    exp_name, s = get_eps_from_logdict_key(sigma)
                    if len(exp_name) > 0:
                        exp_name = '-'+exp_name
                else:
                    s = sigma
                    exp_name = ''
                # if s > 0.125:
                #     continue
                if 'Y' in pnr:
                    y = np.array(pnr['Y'])
                preds = np.array(pnr['Y_pred'])[:max_points]
                radii = np.array(pnr['radii'])[:max_points]
                print(model_name, sigma, preds.shape, radii.shape, y.shape)
                correct = (preds == y[: len(preds)])
                # unique_radii = np.unique(radii)
                # if unique_radii[0] > 0:
                #     unique_radii = np.insert(unique_radii, 0, 0.)
                unique_radii = np.arange(0, radii.max() + radius_step, radius_step)
                
                acc_at_radius = [(correct & (radii >= r)).mean() for r in unique_radii]

                for rad, acc in zip(unique_radii, acc_at_radius):
                    r = {
                        'σ': s,
                        'model_name': f'{model_name}',
                        'radius': rad,
                        'accuracy': acc
                    }
                    data.append(r)
    df = pd.DataFrame(data)
    return df

def plot_cifar10_certified_robustness_results():
    plot_config = OrderedDict([
        ('R-Blur-5FI (σ=0.0625)', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['0.0625', '0.125'])),
        ('R-Blur-5FI (σ=0)', (f'{log_root}/cifar10-0.0/Cifar10RetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', [0.0625, 0.125])),
        ('G-Noise (σ=0.0625)', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseCyclicLRAutoAugmentWideResNet4x22', ['0.0625', '0.125'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseS1250CyclicLRAutoAugmentWideResNet4x22', ['0.125'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 1.75}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', data=df, kind='line')
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.78, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

def plot_ecoset10_certified_robustness_results():
    plot_config = OrderedDict([
        # ('R-Blur-5FI (σ=0.25)', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5Fixation-0.125', '5Fixation-0.25'])),
        ('R-Blur-CFI (σ=0.125)', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('R-Blur-CFI (σ=0)', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
        # ('G-Noise (σ=0.25)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 100)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 1.75}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.78, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

def plot_ecoset100_certified_robustness_results():
    plot_config = OrderedDict([
        ('R-Blur-5FI (σ=0.125)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5Fixation-0.125', '5Fixation-0.25'])),
        ('R-Blur-5FI (σ=0.0)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100RetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5Fixation-0.125', '5Fixation-0.25'])),
        ('R-Blur-CFI (σ=0.125)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
        # ('G-Noise (σ=0.25)', (f'{log_root}/ecoset100-0.0/Ecoset100GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['Centered-0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 100)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.78, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

def plot_ecoset_certified_robustness_results():
    plot_config = OrderedDict([
        # ('R-Warp-CFI', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('R-Blur-CFI (σ=0.125)', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 200)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.78, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

def plot_imagenet_certified_robustness_results():
    plot_config = OrderedDict([
        # ('R-Warp-CFI', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('R-Blur-CFI (σ=0.125)', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', '0.25'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 200)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.78, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))


def plot_ecoset10_pgdinf_atrblur_results():
    plot_config = OrderedDict([
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('AT+R-Blur (σ=0.25)', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainNoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur + AT (1-step)', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur + AT (7-steps)', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrain7StepsRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('R-Blur + AT (Cls Only)', (f'{log_root}/ecoset10-0.008/Ecoset10ClsAdvTrainNoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    ax = sns.catplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df, kind='bar')
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%d')
    # plt.legend([],[], frameon=False)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(11)])
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_atrblur.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset10_pgdl2_atrblur_results():
    plot_config = OrderedDict([
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        # ('AT+R-Blur (σ=0.25)', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainNoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('AT+R-Blur (1-step)', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('AT+R-Blur (7-steps)', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrain7StepsRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2_atrblur.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖2'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)

def plot_ecoset10_pgdinf_results2():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)])
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf-2.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)

def plot_ecoset10_pgdl2_results2():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)])
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2-2.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖2'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)

def plot_ecoset10_pgdinf_vit_results():
    plot_config = OrderedDict([
        ('ViT', (f'{log_root}/ecoset10-0.0/Ecoset10RandAugmentViTCustomSmall', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicRandAugmentViTCustomSmall', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentViTCustomSmall', ['5FixationAPGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainRandAugmentViTCustomSmall', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)])
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_vit.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')

def plot_ecoset10_pgdinf_mlpmixer_results():
    plot_config = OrderedDict([
        ('MLPMixer', (f'{log_root}/ecoset10-0.0/Ecoset10RandAugmentMLPMixerS16', ['APGD'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicRandAugmentMLPMixerS16', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentMLPMixerS16', ['5FixationAPGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainRandAugmentMLPMixerS16', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_mlpmixer.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset10_pgdl2_mlpmixer_results():
    plot_config = OrderedDict([
        ('MLPMixer', (f'{log_root}/ecoset10-0.0/Ecoset10RandAugmentMLPMixerS16', ['APGDL2'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicRandAugmentMLPMixerS16', ['5FixationAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentMLPMixerS16', ['5FixationAPGDL2'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainRandAugmentMLPMixerS16', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2_mlpmixer.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖2'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_cifar10_pgdinf_atrblur_results():
    plot_config = OrderedDict([
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        # ('AT+R-Blur (σ=0.25)', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainNoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('AT+R-Blur (1-step)', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('AT+R-Blur (7-steps)', (f'{log_root}/cifar10-0.008/Cifar10AdvTrain7StepRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_atrblur.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)

def plot_ecoset10_pgdinf_noisestd_results():
    plot_config = OrderedDict([
        ('(σ=0.125)', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        ('(σ=0.25)', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        ('(σ=0.5)', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS5000WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    with sns.plotting_context("paper", font_scale=2.4):
        sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_noisestd.png'), bbox_inches='tight')
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)

def plot_ecoset10_pgdinf_fovarea_results():
    plot_config = OrderedDict([
        (100*(2*15/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        (100*(2*31/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredScale=1APGD'])),
        (100*(2*48/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredScale=2APGD'])),
        (100*(2*64/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredScale=3APGD'])),
        (100*(2*87/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredScale=4APGD'])),
        (100, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredScale=5APGD'])),      
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Proportion of Image at Fovea (%)'
    df = df.rename(columns=colnames, errors="raise")
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.lineplot(x='Proportion of Image at Fovea (%)', y='Accuracy', hue='Perturbation Distance ‖ϵ‖∞', style='Perturbation Distance ‖ϵ‖∞', 
                markers=True, palette=cmap, data=df)
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_fovarea.png'), bbox_inches='tight')
    plt.close()

def plot_ecoset10_pgdinf_beta_results():
    plot_config = OrderedDict([
        (0.001, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500StdScale001WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        (0.01, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500StdScale010WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        (0.025, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500StdScale025WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        (0.05, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        # (100*(2*64/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        # (100*(2*87/224)**2, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        # (100, (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),      
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'β'
    df = df.rename(columns=colnames, errors="raise")
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.lineplot(x='β', y='Accuracy', hue='Perturbation Distance ‖ϵ‖∞', style='Perturbation Distance ‖ϵ‖∞', 
                markers=True, palette=cmap, data=df)
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_beta.png'), bbox_inches='tight')
    plt.close()



def plot_cifar10_cc_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['CCAPGD'])),
        ('R-Warp', (f'{log_root}/cifar10-0.0/Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22', ['5FixationCCAPGD'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationCCAPGD'])),
        ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['CCAPGD'])),
    ])

    df = load_cc_results(plot_config, '/home/mshah1/workhorse3/cifar-10-batches-py/distorted/test_img_ids_and_labels.csv')
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    sns.boxplot(x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, data=df, showfliers=False)
    plt.legend(loc='lower left', ncol=2)
    # plt.legend([],[], frameon=False)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(11)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc.png'))
    plt.close()

    corruption_types = df['Corruption Type'].unique()
    for i, ctype in enumerate(corruption_types):
        sns.boxplot(x='Corruption Severity', y='Accuracy', 
                    hue='Method', hue_order=plot_config, 
                    data=df[df['Corruption Type'] == ctype])
        if i == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
        else:
            plt.legend([],[], frameon=False)
        plt.ylim((0,1))
        plt.yticks([i*10 for i in range(11)])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}.png'))
        plt.close()

def plot_ecoset10_cc_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationCCAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])

    df = load_cc_results(plot_config, '/home/mshah1/workhorse3/ecoset-10/distorted/val_img_paths_and_labels.csv')
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.boxplot(x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, data=df, showfliers=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.legend([],[], frameon=False)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(11)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc.png'))
    plt.close()

    corruption_types = df['Corruption Type'].unique()
    for i, ctype in enumerate(corruption_types):
        sns.boxplot(x='Corruption Severity', y='Accuracy', 
                    hue='Method', hue_order=plot_config, 
                    data=df[df['Corruption Type'] == ctype])
        if i == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
        else:
            plt.legend([],[], frameon=False)
        plt.ylim((0,1))
        plt.yticks([i*10 for i in range(11)])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}.png'))
        plt.close()

def plot_ecoset100_cc_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100CyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianBlurCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])

    df = load_cc_results(plot_config, '/home/mshah1/workhorse3/ecoset-100/distorted/test_img_paths_and_labels.csv')
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    sns.set_style("whitegrid")
    sns.boxplot(x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, data=df, showfliers=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.legend([],[], frameon=False)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(11)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc.png'))
    plt.close()

    corruption_types = df['Corruption Type'].unique()
    for i, ctype in enumerate(corruption_types):
        sns.boxplot(x='Corruption Severity', y='Accuracy', 
                    hue='Method', hue_order=plot_config, 
                    data=df[df['Corruption Type'] == ctype])
        if i == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
        else:
            plt.legend([],[], frameon=False)
        plt.ylim((0,1))
        plt.yticks([i*10 for i in range(11)])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}.png'))
        plt.close()

def plot_ecoset_cc_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])

    df = load_cc_results(plot_config, '/home/mshah1/workhorse3/ecoset/distorted/test_img_paths_and_labels-536K.csv')
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    sns.set_style("whitegrid")
    # sns.catplot(x='Corruption Severity', y='Accuracy', hue='Method', kind="box", col='Corruption Type', hue_order=plot_config, data=df)
    sns.boxplot(x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, data=df, showfliers=False)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.legend([],[], frameon=False)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(11)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc.png'))
    plt.close()

    corruption_types = df['Corruption Type'].unique()
    for i, ctype in enumerate(corruption_types):
        sns.boxplot(x='Corruption Severity', y='Accuracy', 
                    hue='Method', hue_order=plot_config, 
                    data=df[df['Corruption Type'] == ctype])
        if i == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
        else:
            plt.legend([],[], frameon=False)
        plt.ylim((0,1))
        plt.yticks([i/10 for i in range(11)], [i*10 for i in range(11)])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}.png'))
        plt.close()
    
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset/cc_method_plots')
    corruption_types = df['Corruption Method'].unique()
    for i, ctype in enumerate(corruption_types):
        with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
            sns.lineplot(x='Corruption Severity', y='Accuracy', 
                    hue='Method', hue_order=plot_config, 
                    data=df[df['Corruption Method'] == ctype])
        if i == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
        else:
            plt.legend([],[], frameon=False)
        plt.ylim((0,1))
        # plt.yticks([i*10 for i in range(11)])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}.png'))
        plt.close()

def plot_all_ecoset_many_fixation_results():
    plot_config = OrderedDict([
        ('Ecoset-10', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18',[])),
        ('Ecoset-100', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [''])),
        ('Ecoset', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [''])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_many_fixation_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Dataset'
    df = df.rename(columns=colnames, errors="raise")
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    df = df[df['Number of Fixation Points'] == 49]
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        g = sns.catplot(x='Dataset', y='Accuracy', hue='Is Perturbed', kind='bar', data=df, legend=False, aspect=1.5, order=plot_config)
        ax = g.facet_axis(0, 0)
        # legend = ax.legend()
        # legend.texts[0].set_text("")
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_many_fixations.png'), bbox_inches='tight')
    plt.close()

def plot_all_ecoset_five_fixation_results():
    plot_config = OrderedDict([
        ('Ecoset-10', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18',['5FixationAPGD'])),
        ('Ecoset-100', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('Ecoset', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Dataset'
    df = df.rename(columns=colnames, errors="raise")
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.008)]
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        g = sns.catplot(x='Dataset', y='Accuracy', hue='Is Perturbed', kind='bar', data=df, aspect=1.5, legend=False, order=plot_config)
        ax = g.facet_axis(0, 0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_five_fixation.png'), bbox_inches='tight')
    plt.close()

def plot_all_ecoset_AT_results():
    plot_config = OrderedDict([
        ('Ecoset-10', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18',['APGD'])),
        ('Ecoset-100', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('Ecoset', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Dataset'
    df = df.rename(columns=colnames, errors="raise")
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.008)]
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        g = sns.catplot(x='Dataset', y='Accuracy', hue='Is Perturbed', kind='bar', data=df, aspect=1.5, legend=False, order=plot_config)
        ax = g.facet_axis(0, 0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_AT.png'), bbox_inches='tight')
    plt.close()
# def plot_ecoset10_pgdl2_training_ablation_results():
#     plot_config = OrderedDict([
#         ('VDT-5FI', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
#         ('VDT-CFI', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGDL2'])),
#         ('FDT-5FI', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGDL2'])),
#         ('FDT-CFI', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGDL2'])),
#     ])

#     logdicts = get_logdict(plot_config)
#     df = create_data_df(logdicts, plot_config)
#     outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
#     sns.set_style("whitegrid")
#     sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, 'training_ablation_acc_bar_l2.png'))
#     plt.close()

def plot_ecoset10_new_rblur_pgdinf_results():
    new_log_root = '/share/workhorse3/mshah1/biologically_inspired_models/logs'
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Blur-old-5F', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur-new-5F', (f'{new_log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur-old-5F-DN', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationDetNoiseAPGD'])),
        ('R-Blur-new-5F-DN', (f'{new_log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationDetNoiseAPGD'])),
        ('R-Blur-old-CF', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        ('R-Blur-new-CF', (f'{new_log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        ('R-Blur-old-CF-DN', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredDetNoiseAPGD'])),
        ('R-Blur-new-CF-DN', (f'{new_log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredDetNoiseAPGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] <= 0.004])
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_new_rblur_linf.png'))
    plt.close()


# plot_cifar10_pgdinf_results()
# plot_cifar10_pgdl2_results()
# plot_ecoset10_pgdinf_results()
# plot_ecoset10_pgdl2_results()
# plot_ecoset100_pgdinf_results()
# plot_ecoset_pgdinf_results()
# plot_ecoset100_pgdl2_results()
# plot_ecoset100_pgdinf_randaug_results()
# plot_ecoset_pgdl2_results()
# plot_ecoset_pgdinf_results_with_affine()
# plot_imagenet_pgdinf_results()
# plot_imagenet_pgdl2_results()
# plot_imagenet_pgd_results()

# plot_cifar10_pgdinf_training_ablation_results()
# plot_cifar10_pgdl2_training_ablation_results()

# plot_ecoset10_pgdinf_training_ablation_results_1()
# plot_ecoset10_pgdinf_training_ablation_results_2()
# plot_ecoset10_pgdl2_training_ablation_results()

# plot_cifar10_certified_robustness_results()
# plot_ecoset10_certified_robustness_results()
plot_ecoset100_certified_robustness_results()
# plot_ecoset_certified_robustness_results()
# plot_imagenet_certified_robustness_results()

# plot_ecoset10_pgdinf_atrblur_results()
# plot_ecoset10_pgdl2_atrblur_results()
# plot_cifar10_pgdinf_atrblur_results()

# plot_ecoset10_pgdinf_results2()
# plot_ecoset10_pgdinf_vit_results()
# plot_ecoset10_pgdinf_mlpmixer_results()

# plot_ecoset10_pgdl2_results2()
# plot_ecoset10_pgdl2_mlpmixer_results()

# plot_ecoset10_pgdinf_noisestd_results()
# plot_ecoset10_pgdinf_fovarea_results()
# plot_ecoset10_pgdinf_beta_results()

# plot_cifar10_cc_results()
# plot_ecoset10_cc_results()
# plot_ecoset100_cc_results()
# plot_ecoset_cc_results()

# plot_ecoset10_new_rblur_pgdinf_results()
# plot_all_ecoset_many_fixation_results()
# plot_all_ecoset_five_fixation_results()

# plot_all_ecoset_AT_results()

# def foo():
#     plot_config = OrderedDict([
#         ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
#         ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
#     ])
#     logdicts = get_logdict(plot_config)
#     ecoset_df = create_data_df(logdicts, plot_config)

#     plot_config = OrderedDict([
#         ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGD', 'APGDL2'])),
#         ('R-Warp', (f'{log_root}/cifar10-0.0/Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['APGD', 'APGDL2'])),
#     ])
#     logdicts = get_logdict(plot_config)
#     cifar10_df = create_data_df(logdicts, plot_config)

#     plot_config = OrderedDict([
#         ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
#         ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
#     ])
#     logdicts = get_logdict(plot_config)
#     ecost10_df = create_data_df(logdicts, plot_config)

#     plot_config = OrderedDict([
#         ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
#         ('R-Warp', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD', '5FixationAPGDL2'])),
#         ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
#     ])
#     logdicts = get_logdict(plot_config)
#     ecoset100_df = create_data_df(logdicts, plot_config)
    
#     low_eps = [0.002, 0.004]
#     high_eps = [0.008, 0.016]
#     for df, name in zip([cifar10_df, ecost10_df, ecoset100_df, ecoset_df], ['c10', 'e10', 'e100', 'e']):
#         print(name)
#         for method in df['Method'].unique():
#             lo_acc_diff1 = df[(df['Method'] == 'R-Blur') & (df['Perturbation Distance ‖ϵ‖∞'] == low_eps[0])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == low_eps[0])]['Accuracy'].values.mean()
#             lo_acc_diff2 = df[(df['Method'] == 'R-Blur') & (df['Perturbation Distance ‖ϵ‖∞'] == low_eps[1])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == low_eps[1])]['Accuracy'].values.mean()
#             avg_lo_acc_diff = (lo_acc_diff1+lo_acc_diff2)/2

#             hi_acc_diff1 = df[(df['Method'] == 'R-Blur') & (df['Perturbation Distance ‖ϵ‖∞'] == high_eps[0])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == high_eps[0])]['Accuracy'].values.mean()
#             hi_acc_diff2 = df[(df['Method'] == 'R-Blur') & (df['Perturbation Distance ‖ϵ‖∞'] == high_eps[1])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == high_eps[1])]['Accuracy'].values.mean()
#             avg_hi_acc_diff = (hi_acc_diff1+hi_acc_diff2)/2
#             print(method, (lo_acc_diff1, lo_acc_diff2), (hi_acc_diff1, hi_acc_diff2), avg_lo_acc_diff, avg_hi_acc_diff)

# def bar():
#     plot_config = OrderedDict([
#         ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['CCAPGD'])),
#         ('R-Warp', (f'{log_root}/cifar10-0.0/Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22', ['5FixationCCAPGD'])),
#         ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationCCAPGD'])),
#         ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['CCAPGD'])),
#     ])
#     cifar10_df = load_cc_results(plot_config, '/home/mshah1/workhorse3/cifar-10-batches-py/distorted/test_img_ids_and_labels.csv')

#     plot_config = OrderedDict([
#             ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#             # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
#             ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
#             ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationCCAPGD'])),
#             # ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
#             ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#         ])
#     ecoset10_df = load_cc_results(plot_config, '/home/mshah1/workhorse3/ecoset-10/distorted/val_img_paths_and_labels.csv')

#     plot_config = OrderedDict([
#             ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100CyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#             # ('GBlur (σ=10.5)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianBlurCyclicLRRandAugmentXResNet2x18', ['APGD'])),
#             ('R-Warp', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
#             ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
#             # ('G-Noise', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#             ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#         ])
#     ecoset100_df = load_cc_results(plot_config, '/home/mshah1/workhorse3/ecoset-100/distorted/test_img_paths_and_labels.csv')

#     plot_config = OrderedDict([
#             ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#             ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
#             ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationCCAPGD'])),
#             # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#             ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
#         ])
#     ecoset_df = load_cc_results(plot_config, '/home/mshah1/workhorse3/ecoset/distorted/test_img_paths_and_labels-536K.csv')

#     low_eps = [1,2]
#     high_eps = [4,5]
#     for df, name in zip([cifar10_df, ecoset10_df, ecoset100_df, ecoset_df], ['c10', 'e10', 'e100', 'e']):
#         print(name)
#         for method in df['Method'].unique():
#             lo_acc_diff1 = df[(df['Method'] == 'R-Blur') & (df['Corruption Severity'] == low_eps[0])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Corruption Severity'] == low_eps[0])]['Accuracy'].values.mean()
#             lo_acc_diff2 = df[(df['Method'] == 'R-Blur') & (df['Corruption Severity'] == low_eps[1])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Corruption Severity'] == low_eps[1])]['Accuracy'].values.mean()
#             avg_lo_acc_diff = (lo_acc_diff1+lo_acc_diff2)/2

#             hi_acc_diff1 = df[(df['Method'] == 'R-Blur') & (df['Corruption Severity'] == high_eps[0])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Corruption Severity'] == high_eps[0])]['Accuracy'].values.mean()
#             hi_acc_diff2 = df[(df['Method'] == 'R-Blur') & (df['Corruption Severity'] == high_eps[1])]['Accuracy'].values.mean() - df[(df['Method'] == method) & (df['Corruption Severity'] == high_eps[1])]['Accuracy'].values.mean()
#             avg_hi_acc_diff = (hi_acc_diff1+hi_acc_diff2)/2
#             print(method, avg_lo_acc_diff, avg_hi_acc_diff)
# # foo()
# bar()
#     # Perturbation Distance ‖ϵ‖∞
#     # Perturbation Distance ‖ϵ‖2