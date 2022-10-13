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
from adversarialML.biologically_inspired_models.src.utils import _compute_area_under_curve, load_logs, get_eps_from_logdict_key
from adversarialML.biologically_inspired_models.src.ICLR22.write_all_results_to_csv import create_all_metrics_df
import re

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
        data_and_preds = logdict['adv_data_and_preds']
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
                            'Accuracy': a,
                            'Attack': atkname
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
    print(logdirs)
    logdict = load_logs(logdirs, labels)
    return logdict

def maybe_create_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d

log_root = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs'
outdir_root = 'ICLR22/visualizations'
sns.set(font_scale=1.25)

def plot_cifar10_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        # ('GBlur (σ=1.5)', (f'{log_root}/cifar10-0.0/Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('R-Warp', (f'{log_root}/cifar10-0.0/Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
        ('G-Noise', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
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
        ('G-Noise', (f'{log_root}/cifar10-0.0/Cifar10GaussianNoiseCyclicLRAutoAugmentWideResNet4x22', ['APGDL2'])),
        ('AT', (f'{log_root}/cifar10-0.008/Cifar10AdvTrainCyclicLRAutoAugmentWideResNet4x22', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_ecoset10_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('GBlur (σ=10.5)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('R-Warp', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
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
        ('G-Noise', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖∞'] > 0.])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
    plt.close()

    clean_results_df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0]
    clean_results_df = pd.pivot_table(clean_results_df, values='Accuracy', index='Method')
    print(clean_results_df)
    clean_results_df.to_csv(os.path.join(outdir, 'clean_accuracy.csv'))

def plot_ecoset_pgdinf_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGD'])),
        # ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    print(df)
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf.png'))
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
        ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
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
        ('G-Noise', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT', (f'{log_root}/ecoset100_folder-0.008/Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, estimator=max, ci=None, data=df[df['Perturbation Distance ‖ϵ‖2'] > 0.])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
    plt.close()

def plot_ecoset_pgdl2_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('R-Blur (σ=0.25)', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        # ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['5FixationAPGDL2'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
        ('AT (‖ϵ‖∞=0.008)', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGDL2'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    print(df)
    sns.set_style("whitegrid")
    sns.barplot(x='Perturbation Distance ‖ϵ‖2', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_l2.png'))
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

def plot_ecoset10_pgdinf_training_ablation_results():
    plot_config = OrderedDict([
        ('VDT-5FI', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('VDT-CFI', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
        ('FDT-5FI', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('NoNoise', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('OnlyColor', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500OnlyColorWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('NoBlur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500NoBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        ('G-Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('GBlur-WNoise', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('GBlur-NoNoise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        # ('FDT-CFI', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurCyclicLR1e_1RandAugmentXResNet2x18', ['CenteredAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    plot = sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config,
                # data=df[(df['Perturbation Distance ‖ϵ‖∞'] == 0.) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.008)]
                data=df[dataframe_or(df, 'Perturbation Distance ‖ϵ‖∞', [0., .008])]
                )
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_ablation_acc_bar_linf.png'))
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
        ('R-Blur-5FI (σ=0.25)', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5Fixation-0.125', '5Fixation-0.25'])),
        ('R-Blur-CFI (σ=0)', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('G-Noise (σ=0.125)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
        ('G-Noise (σ=0.25)', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['0.25'])),
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
        ('G-Noise (σ=0.125)', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100GaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 100)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 1.75}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', row='σ', hue_order=plot_config.keys(), data=df, kind='line')
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.78, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

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

# plot_cifar10_pgdinf_results()
# plot_cifar10_pgdl2_results()
# plot_ecoset10_pgdinf_results()
# plot_ecoset10_pgdl2_results()
# plot_ecoset100_pgdinf_results()
# plot_ecoset_pgdinf_results()
# plot_ecoset100_pgdl2_results()
# plot_ecoset_pgdl2_results()

# plot_cifar10_pgdinf_training_ablation_results()
# plot_cifar10_pgdl2_training_ablation_results()

plot_ecoset10_pgdinf_training_ablation_results()
# plot_ecoset10_pgdl2_training_ablation_results()

# plot_cifar10_certified_robustness_results()
# plot_ecoset10_certified_robustness_results()
# plot_ecoset100_certified_robustness_results()