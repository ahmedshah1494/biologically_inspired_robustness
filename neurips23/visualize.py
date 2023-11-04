from collections import OrderedDict
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rblur.utils import get_eps_from_logdict_key, load_json, aggregate_dicts, lazy_load_pickle
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
            norm = 2 if (("L2" in atkname) or (float(eps) >= 0.1)) else "∞"
            if ((norm == 2) and (eps <= 2.5)) or (eps <= 0.016):
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
                            f'Perturbation Distance ‖ϵ‖{norm}': float(eps),
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
            is_worstcase = atkstr.startswith('worstcase')
            eps = float(epsstr.split('=')[-1])
            npoints = int(nstr.split('=')[-1])
            # accs = [accs[best_model_idx]]
            for i,a in enumerate(accs):
                r = {
                    'Method': model_name,
                    f'Perturbation Distance ‖ϵ‖∞': float(eps),
                    'Accuracy': a*100,
                    'Number of Fixation Points': npoints,
                    'is_worstcase': is_worstcase
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
        print(metric_files)
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
        print(lnp_pth, path_and_label_file, 'len(cns_list)=',len(cns_list), 'len(is_correct)=',len(is_correct))
        assert len(cns_list) == len(is_correct)
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
    def get_pred_and_labels_files(d, atk):
        fp1 = os.path.join(d, 'per_attack_results', f'{atk}-0.0_label_and_preds.csv')
        # fp2 = os.path.join(d, 'per_attack_results', f'{atk}-0.0_label_and_preds_2.csv')
        if os.path.exists(fp1):
            return fp1
        # elif os.path.exists(fp2):
        #     return fp2

    with open(path_and_label_file) as f:
        fnames = [l.split(',')[-1].split('/')[-1].split('.')[0] for l in f.readlines()]
        corruption_and_severity = [fn[:fn.index('-')+2].split('-') for fn in fnames]
        corruption_and_severity = [(x[0],int(x[1])) for x in corruption_and_severity]
    logdirs_and_labels = [(ld, label, atk[0]) for label, (ld, atk) in plot_config.items()]
    logdict = {}
    for logdir, label, atk in logdirs_and_labels:
        expdirs = [os.path.join(logdir, x) for x in os.listdir(logdir)]
        metric_files = [get_pred_and_labels_files(d, atk) for d in expdirs]
        metric_files = [x for x in metric_files if x is not None]
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
                        '$\sigma_c$': s,
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
        # ('R-Blur-CFI (σ=0.125)', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        # ('G-Noise (σ=0.125)', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
        # ('G-Noise($\sigma_t=0.25$)', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
        ('R-Blur-5FI', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5Fixation-0.125', '5Fixation-0.25'])),
        ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
        ('R-Blur-CFI', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        # ('R-Blur-5FI-N (σ=0.125)', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['DetNoise5Fixation-0.125'])),
        ('R-Blur-5FI\n($\sigma_t=0.0$)', (f'{log_root}/ecoset-0.0/EcosetRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5Fixation-0.125'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 200)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', col='$\sigma_c$', hue_order=plot_config.keys(), data=df, kind='line')
        plot._legend.set_title("")
        sns.move_legend(plot, "upper center", bbox_to_anchor=(0.82, 0.92))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line_2.pdf'))
    plt.close('all')

    for sig in df['$\sigma_c$'].unique():
        # fig = plt.figure(figsize=(5,6))
        with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 2.}):
            sns.set_style("whitegrid")
            plot=sns.lineplot(x='radius', y='accuracy', hue='model_name', hue_order=plot_config.keys(), data=df[df['$\sigma_c$']==sig])
        plt.legend(title='', fontsize=17)
            # sns.move_legend(plot, "right", bbox_to_anchor=(1.5, 0.92))
        plt.tight_layout()
        plt.title('Ecoset')
        plt.savefig(os.path.join(outdir, f'rs_certified_acc_line_2_{sig}.pdf'))
        plt.close('all')

def plot_imagenet_certified_robustness_results():
    plot_config = OrderedDict([
        # ('R-Warp-CFI', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Centered-0.125', 'Centered-0.25'])),
        ('R-Blur-5FI', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5Fixation-0.125', '5Fixation-0.25'])),
        ('G-Noise', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['0.125', '0.25'])),
    ])
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    logdicts = get_logdict(plot_config)
    df = create_rs_dataframe(logdicts, plot_config, 200)
    plt.figure(figsize=(5,8))
    with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 2.}):
        sns.set_style("whitegrid")
        plot=sns.relplot(x='radius', y='accuracy', hue='model_name', col='$\sigma_c$', hue_order=plot_config.keys(), data=df, kind='line')
        plot._legend.remove()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.pdf'))
    plt.close('all')

    for sig in df['$\sigma_c$'].unique():
        # fig = plt.figure(figsize=(5,6))
        with sns.plotting_context("paper", font_scale=2.4, rc={'lines.linewidth': 2.}):
            sns.set_style("whitegrid")
            plot=sns.lineplot(x='radius', y='accuracy', hue='model_name', hue_order=plot_config.keys(), data=df[df['$\sigma_c$']==sig])
        plt.legend(title='', fontsize=17)
            # sns.move_legend(plot, "right", bbox_to_anchor=(1.5, 0.92))
        plt.tight_layout()
        plt.title('Imagenet')
        plt.savefig(os.path.join(outdir, f'rs_certified_acc_line_2_{sig}.pdf'))
        plt.close('all')


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

def plot_neurips_ecoset_cc_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CC5RandAugAPGD'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsDetNoisedeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseCCAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])
    
    _plot_config = copy.deepcopy(plot_config)
    plot_config2 = {}
    plot_config2['AT'] = _plot_config.pop('AT')
    plot_config2['ResNet'] = _plot_config.pop('ResNet')

    df1 = load_cc_results(_plot_config, '/home/mshah1/workhorse3/ecoset/distorted/test_img_paths_and_labels-268K.csv')
    # print(df1['Method'].unique())
    df2 = load_cc_results(plot_config2, '/home/mshah1/workhorse3/ecoset/distorted/test_img_paths_and_labels-536K.csv')
    # print(df2['Method'].unique())
    df = pd.concat([df1, df2]).reset_index(drop=True)
    print(df)
    # for m in df['Method'].unique():
    #     print(m, len(df[df['Method'] == m]), df[df['Method'] == m]['Corruption Method'].unique(), df[df['Method'] == m]['Corruption Severity'].unique())
    # exit()
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    sns.set_style("whitegrid")
    # sns.catplot(x='Corruption Severity', y='Accuracy', hue='Method', kind="box", col='Corruption Type', hue_order=plot_config, data=df)
    # sns.boxplot(x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, data=df, showfliers=False)
    # # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    # # plt.legend([],[], frameon=False)
    # # plt.ylim((0,1))
    # # plt.yticks([i*10 for i in range(11)])
    # plt.tight_layout()
    # plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc_nips.pdf'))
    # plt.close()

    df['Accuracy'] = df['Accuracy'] * 100
    plt.figure(figsize=(12,5))
    with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 3.}):
        ax=sns.barplot(data=df, x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, ci=None)
        ax.set_ylim((0,60))
    with sns.plotting_context("paper", font_scale=2., rc={'lines.linewidth': 3.}):
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', rotation=90)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc.pdf'))
    plt.close()

    # corruption_types = df['Corruption Type'].unique()
    # for i, ctype in enumerate(corruption_types):
    #     sns.boxplot(x='Corruption Severity', y='Accuracy', 
    #                 hue='Method', hue_order=plot_config, 
    #                 data=df[df['Corruption Type'] == ctype])
    #     if i == 0:
    #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
    #     else:
    #         plt.legend([],[], frameon=False)
    #     plt.ylim((0,1))
    #     plt.yticks([i/10 for i in range(11)], [i*10 for i in range(11)])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}_nips.pdf'))
    #     plt.close()
    
    # outdir = maybe_create_dir(f'{outdir_root}/Ecoset/cc_method_plots')
    # corruption_types = df['Corruption Method'].unique()
    # for i, ctype in enumerate(corruption_types):
    #     with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
    #         sns.lineplot(x='Corruption Severity', y='Accuracy', 
    #                 hue='Method', hue_order=plot_config, 
    #                 data=df[df['Corruption Method'] == ctype])
    #     if i == 0:
    #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
    #     else:
    #         plt.legend([],[], frameon=False)
    #     plt.ylim((0,1))
    #     # plt.yticks([i*10 for i in range(11)])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}_nips.pdf'))
    #     plt.close()

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

def _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=False, min_eps=0., max_eps=1., legend=False, figsize=(8,4)):
    xlabel = f'Perturbation Distance ‖ϵ‖{norm}'
    ylabel = 'Accuracy'
    sns.set_style("whitegrid")
    colors = sns.utils.get_color_cycle()[:len(plot_config)]
    methods = list(plot_config.keys())
    plt.figure(figsize=figsize)
    with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 2.}):
        if stacked:
            for method, c in zip(methods[::-1], colors[::-1]):
                ax = sns.barplot(x=xlabel, y='Accuracy', color=c, data=df[(df['Method'] == method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
        else:
            ax = sns.barplot(x=xlabel, y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
    
        for method, c in zip(plot_config, colors):
            ax.axhline(df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]['Accuracy'].mean(), color=c, linestyle='--')
        for container in ax.containers:
            if stacked:
                container.datavalues = np.array([v if v > 1 else np.nan for v in container.datavalues])
            ax.bar_label(container, fmt='%d', label_type='edge')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    if not legend:
        plt.legend([],[], frameon=False)
    plt.tight_layout()
    return ax

def plot_cifar10_baseline_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGD'])),
        ('5-RandAffine', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD'])),
    ])
    xlabel = 'Perturbation Distance ‖ϵ‖∞'
    ylabel = 'Accuracy'
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_linf{"_stacked" if stacked else ""}.pdf'))
    plt.close()



def plot_cifar10_baseline_pgdl2_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGD','APGDL2'])),
        ('5-RandAffine', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD','5FixationAPGDL2'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    outdir = maybe_create_dir(f'{outdir_root}/cifar10')
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset10_baseline_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('5-RandAffine', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_linf{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset10_baseline_pgdl2_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', [
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.5]
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()
    
def plot_ecoset100_baseline_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('5-RandAffine', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100CyclicLRRandAugmentXResNet2x18', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_linf{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset100_baseline_pgdl2_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100CyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.5]
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset_baseline_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    sns.set_style("whitegrid")
    _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_linf{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset_baseline_pgdl2_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGDL2'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.5]
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_baseline_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGD'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_linf{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_baseline_pgdl2_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGDL2'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.5]
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def _plot_baseline_pgd_all_results(plot_config, legend):
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)

    df['perturbation size'] = 'n/a'
    df.loc[df['Perturbation Distance ‖ϵ‖2'] == 0., 'perturbation size'] = 'none'
    df.loc[df['Perturbation Distance ‖ϵ‖2'] == .5, 'perturbation size'] = 'small'
    df.loc[df['Perturbation Distance ‖ϵ‖2'] == 1., 'perturbation size'] = 'moderate'
    df.loc[df['Perturbation Distance ‖ϵ‖2'] == 2.0, 'perturbation size'] = 'large'
    df.loc[df['Perturbation Distance ‖ϵ‖2'] > 2.0, 'perturbation size'] = 'xlarge'

    df.loc[df['Perturbation Distance ‖ϵ‖∞'] == 0., 'perturbation size'] = 'none'
    df.loc[df['Perturbation Distance ‖ϵ‖∞'] == .002, 'perturbation size'] = 'small'
    df.loc[df['Perturbation Distance ‖ϵ‖∞'] == .004, 'perturbation size'] = 'moderate'
    df.loc[df['Perturbation Distance ‖ϵ‖∞'] == .008, 'perturbation size'] = 'large'
    df.loc[df['Perturbation Distance ‖ϵ‖∞'] > .008, 'perturbation size'] = 'xlarge'

    print(df)
    sns.set_style("whitegrid")
    colors = sns.utils.get_color_cycle()[:len(plot_config)]
    methods = list(plot_config.keys())
    ax = sns.barplot(x='perturbation size', y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df['perturbation size'] != 'n/a') & (df['perturbation size'] != 'none') & (df['perturbation size'] != 'xlarge')])
    for method, c in zip(plot_config, colors):
        ax.axhline(df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]['Accuracy'].mean(), color=c, linestyle='--')
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    if not legend:
        plt.legend([],[], frameon=False)
    else:
        plt.legend(loc='upper right')

def plot_ecoset100_pgd_all_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset100')
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100CyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100CyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGD', '5RandAugAPGDL2'])),
        ('R-Warp', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100RetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100VOneBlockCyclicLRRandAugmentXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/ecoset100_folder-0.0/ecoset100_folder-0.0//Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('AT', (f'{log_root}/ecoset100_folder-0.008//Ecoset100AdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
    ])
    _plot_baseline_pgd_all_results(plot_config, legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar.pdf'), format='pdf')
    plt.close()

def plot_ecoset_pgd_all_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGD', ''])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
    ])
    _plot_baseline_pgd_all_results(plot_config, legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar.pdf'), format='pdf')
    plt.close()

def plot_imagenet_pgd_all_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGD', ''])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
    ])
    _plot_baseline_pgd_all_results(plot_config, legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar.pdf'), format='pdf')
    plt.close()

def plot_imagenet_baseline_pgd_all_results(stacked=False, min_eps=0., max_eps=2.5, legend=False):
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])
    _plot_baseline_pgd_all_results(plot_config, legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar_baseline.pdf'), format='pdf')
    plt.close()

def _plot_delta_pgd_results(plot_config, norm, stacked, min_eps, max_eps, legend):
    if norm == 'inf':
        xlabel = 'Perturbation Distance ‖ϵ‖∞'
        eps = [.002, .004, .008]
    if norm == 2:
        xlabel = 'Perturbation Distance ‖ϵ‖2'
        eps = [.5, 1., 2.]

    
    

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.5]

    delta_col_name = '$\Delta Accuracy (\%)$'
    df[delta_col_name] = None
    baseline_method = 'R-Blur'

    for e in eps:
        for method in plot_config:
            if method == baseline_method:
                continue
            df.loc[(df['Method'] == method) & (df[xlabel] == e), delta_col_name] = (df[(df['Method'] == baseline_method) & (df[xlabel] == e)]['Accuracy'].values - df[(df['Method'] == method) & (df[xlabel] == e)]['Accuracy'].values)

    # print(df[(df['Method'] == baseline_method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)].sort_values(xlabel))
    # rblur_acc = df[(df['Method'] == baseline_method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)].sort_values(xlabel)['Accuracy'].values
    # for method in plot_config:
    #         print(df[(df['Method'] == method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)].sort_values(xlabel))
    #         method_acc = df[(df['Method'] == method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)].sort_values(xlabel)['Accuracy'].values
    #         df.loc[(df['Method'] == method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps), delta_col_name] = (rblur_acc - method_acc)
    # print(df)

    sns.set_style("whitegrid")
    colors = sns.utils.get_color_cycle()[3:len(plot_config)]
    methods = list(plot_config.keys())
    
    df = df[(df['Method'] != baseline_method)]
    methods.remove(baseline_method)
    plt.figure(figsize=(8,4))
    with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 2.}):
        if stacked:
            for method, c in zip(methods[::-1], colors[::-1]):
                ax = sns.barplot(x=xlabel, y=delta_col_name, color=c, data=df[(df['Method'] == method) & dataframe_or(df, xlabel, eps)])
        else:
            print(df[(df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
            ax = sns.barplot(x=xlabel, y=delta_col_name, hue='Method', hue_order=methods, data=df[dataframe_or(df, xlabel, eps)])
        # for method, c in zip(plot_config, colors):
        #     ax.axhline(df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]['Accuracy'].mean(), color=c, linestyle='--')
        for container in ax.containers:
            if stacked:
                container.datavalues = np.array([v if v > 1 else np.nan for v in container.datavalues])
            ax.bar_label(container, fmt='%+d')
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    if not legend:
        plt.legend([],[], frameon=False)
    else:
        plt.legend(loc='upper right')

def plot_ecoset_biomodels_delta_pgd_results(norm='inf', stacked=False, min_eps=0., max_eps=2.5, legend=False):
    plot_config = OrderedDict([
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])

    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    _plot_delta_pgd_results(plot_config, norm, stacked, min_eps, max_eps, legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar_biomodels_delta_l{norm}{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_biomodels_delta_pgd_results(norm='inf', stacked=False, min_eps=0., max_eps=2.5, legend=False):
    plot_config = OrderedDict([
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])
    
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    _plot_delta_pgd_results(plot_config, norm, stacked, min_eps, max_eps, legend)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar_biomodels_delta_l{norm}{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset_biomodels_pgd_results(norm='inf', stacked=False, min_eps=0., max_eps=2.5, legend=False):
    if norm == 'inf':
        xlabel = 'Perturbation Distance ‖ϵ‖∞'
    if norm == 2:
        xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset')
    sns.set_style("whitegrid")
    colors = sns.utils.get_color_cycle()[:len(plot_config)]
    methods = list(plot_config.keys())
    if stacked:
        for method, c in zip(methods[::-1], colors[::-1]):
            ax = sns.barplot(x=xlabel, y='Accuracy', color=c, data=df[(df['Method'] == method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
    else:
        ax = sns.barplot(x=xlabel, y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
    for method, c in zip(plot_config, colors):
        ax.axhline(df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]['Accuracy'].mean(), color=c, linestyle='--')
    for container in ax.containers:
        if stacked:
            container.datavalues = np.array([v if v > 1 else np.nan for v in container.datavalues])
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    if not legend:
        plt.legend([],[], frameon=False)
    else:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar_biomodels_l{norm}{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_biomodels_pgd_results(norm='inf', stacked=False, min_eps=0., max_eps=2.5, legend=False):
    if norm == 'inf':
        xlabel = 'Perturbation Distance ‖ϵ‖∞'
    if norm == 2:
        xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.]
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    sns.set_style("whitegrid")
    colors = sns.utils.get_color_cycle()[:len(plot_config)]
    methods = list(plot_config.keys())
    if stacked:
        for method, c in zip(methods[::-1], colors[::-1]):
            ax = sns.barplot(x=xlabel, y='Accuracy', color=c, data=df[(df['Method'] == method) & (df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
    else:
        ax = sns.barplot(x=xlabel, y='Accuracy', hue='Method', hue_order=plot_config, data=df[(df[xlabel] >= min_eps) & (df[xlabel] <= max_eps)])
    for method, c in zip(plot_config, colors):
        ax.axhline(df[(df['Method'] == method) & (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]['Accuracy'].mean(), color=c, linestyle='--')
    for container in ax.containers:
        if stacked:
            container.datavalues = np.array([v if v > 1 else np.nan for v in container.datavalues])
        ax.bar_label(container, fmt='%d')
    plt.ylim((0,1))
    plt.yticks([i*10 for i in range(11)], [i*10 for i in range(11)])
    if not legend:
        plt.legend([],[], frameon=False)
    else:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_acc_bar_biomodels_l{norm}{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()



def plot_imagenet_cc_results():
    plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['CC5RandAugAPGD'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsDetNoisedeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
        ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseCCAPGD'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])

    df = load_cc_results(plot_config, '/home/mshah1/workhorse3/imagenet/distorted/test_img_paths_and_labels-190K.csv')
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')


    sns.set_style("whitegrid")
    # # sns.catplot(x='Corruption Severity', y='Accuracy', hue='Method', kind="box", col='Corruption Type', hue_order=plot_config, data=df)
    # sns.boxplot(x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, data=df, showfliers=False)
    # # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    # plt.legend([],[], frameon=False)
    # # plt.ylim((0,1))
    # # plt.yticks([i*10 for i in range(11)])
    # plt.tight_layout()
    # plt.savefig(os.path.join(outdir, 'test_acc_box_allcc.pdf'))
    # plt.close()

    # ce_rows = []
    # base_model_df = df[df['Method'] == 'ResNet']
    # for corr in df['Corruption Method'].unique():
    #     base_acc = base_model_df[base_model_df['Corruption Method'] == corr]['Accuracy'].mean()
    #     for model_name in df['Method'].unique():
    #         model_acc = df[(df['Method'] == model_name) & (df['Corruption Method'] == corr)]['Accuracy'].mean()
    #         ce = model_acc / base_acc
    #         ce_rows.append({
    #             'Method': model_name,
    #             'Corruption Method': corr,
    #             'mCE': ce
    #         })
    # ce_df = pd.DataFrame(ce_rows)
    # print(ce_df)
    # plt.figure(figsize=(8,8))
    # with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 3.}):
    #     sns.barplot(data=ce_df, x='Method', y='mCE', hue='Method', hue_order=plot_config, ci=None)
    # plt.legend([],[], frameon=False)
    # plt.xticks(rotation=45, ha='center')
    # plt.tight_layout()
    # plt.savefig(os.path.join(outdir, 'test_mce_bar_allcc.pdf'))
    # plt.close()

    df['Accuracy'] = df['Accuracy'] * 100
    plt.figure(figsize=(12,5.25))
    with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 3.}):
        ax=sns.barplot(data=df, x='Corruption Severity', y='Accuracy', hue='Method', hue_order=plot_config, ci=None)
        ax.set_ylim((0,60))
    with sns.plotting_context("paper", font_scale=2., rc={'lines.linewidth': 3.}):
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', rotation=90)
        # plt.legend([],[], frameon=False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.03), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_allcc.pdf'))
    plt.close()
            
    # plt.figure(figsize=(8,8))
    # with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 3.}):
    #     sns.lineplot(data=df, x='Corruption Severity', y='Accuracy', hue='Method', style='Method', hue_order=plot_config, ci=None)
    #     # plt.legend([],[], frameon=False)
    # plt.xticks(range(1,6))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(outdir, 'test_acc_line_allcc.pdf'))
    # plt.close()

    # corruption_types = df['Corruption Type'].unique()
    # for i, ctype in enumerate(corruption_types):
    #     sns.boxplot(x='Corruption Severity', y='Accuracy', 
    #                 hue='Method', hue_order=plot_config, 
    #                 data=df[df['Corruption Type'] == ctype])
    #     if i == 0:
    #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
    #     else:
    #         plt.legend([],[], frameon=False)
    #     plt.ylim((0,1))
    #     plt.yticks([i/10 for i in range(11)], [i*10 for i in range(11)])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outdir, f'test_acc_bar_{ctype}.pdf'))
    #     plt.close()
    
    # outdir = maybe_create_dir(f'{outdir_root}/Imagenet/cc_method_plots')
    # corruption_types = df['Corruption Method'].unique()
    # for i, ctype in enumerate(corruption_types):
    #     with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
    #         sns.lineplot(x='Corruption Severity', y='Accuracy', 
    #                 hue='Method', hue_order=plot_config, 
    #                 data=df[df['Corruption Method'] == ctype])
    #     if i == 0:
    #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=2)
    #     else:
    #         plt.legend([],[], frameon=False)
    #     plt.ylim((0,1))
    #     # plt.yticks([i*10 for i in range(11)])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outdir, f'test_acc_box_{ctype}.pdf'))
    #     plt.close()

def create_cc_table_df(logdicts, plot_config):
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
            eps = float(eps)
            norm = 2 if (("L2" in atkname) or (float(eps) >= 0.1)) else "∞"
            atktype = 'corruption' if 'CC' in atkname else 'whitebox'
            atktype = 'clean' if ((atktype == 'whitebox') and (eps == 0.)) else atktype
            if ((norm == 2) and (eps <= 2.5)) or (eps <= 0.016):
                if atkname in metrics_to_plot:
                    # accs = [accs[best_model_idx]]
                    for i,a in enumerate(accs):
                        r = {
                            'Method': model_name,
                            f'Perturbation Distance ‖ϵ‖{norm}': eps,
                            'Perturbation Type': atktype,
                            'Accuracy': a*100,
                            'Attack': atkname
                        }
                        data.append(r)
    df = pd.DataFrame(data)
    return df


def plot_ecoset_cc_results_table(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    log_root = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs'
    cc_plot_config = OrderedDict([
            ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CC5RandAugAPGD'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsDetNoisedeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseCCAPGD'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
        # ('G-Noise', (f'{log_root}/ecoset-0.0/EcosetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])
    wb_plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGDL2_25'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
    ])
    all_plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CCAPGD', 'APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/ecoset-0.0/EcosetCyclicLRRandAugmentXResNet2x18', ['CC5RandAug', 'CC5RandAugAPGD', '5RandAug'])),
        ('R-Blur', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('R-Warp', (f'{log_root}/ecoset-0.0/EcosetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsDetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD',
                                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGD_25',
                                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGDL2_25'])),
        ('VOneBlock', (f'{log_root}/ecoset-0.0/EcosetVOneBlockCyclicLRXResNet2x18', ['DetNoiseCCAPGD', 'DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('AT', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD', 'APGD', 'APGDL2'])),
    ])
    cc_plot_config2 = {}
    cc_plot_config2['AT'] = cc_plot_config.pop('AT')
    cc_plot_config2['ResNet'] = cc_plot_config.pop('ResNet')

    cc_df1 = load_cc_results(cc_plot_config, '/home/mshah1/workhorse3/ecoset/distorted/test_img_paths_and_labels-268K.csv')
    cc_df2 = load_cc_results(cc_plot_config2, '/home/mshah1/workhorse3/ecoset/distorted/test_img_paths_and_labels-536K.csv')
    cc_df = pd.concat([cc_df1, cc_df2])

    logdicts = get_logdict(wb_plot_config)
    wb_df = create_cc_table_df(logdicts, wb_plot_config)

    hisev_cc_df = cc_df[cc_df['Corruption Severity'] > 3]
    losev_cc_df = cc_df[cc_df['Corruption Severity'] <= 3]
    filtered_cc_df = cc_df[(cc_df['Corruption Method'] != 'gaussian_noise') & (cc_df['Corruption Method'] != 'gaussian_blur')]
    hisev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] > 3]
    losev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] <= 3]

    wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [0., .002, .004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5, 1.5, 2.0])]
    hi_wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [.004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [1.5, 2.0])]
    lo_wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [.002]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5])]

    result_rows = []
    for method in wb_df['Method'].unique():
        hi_wbdf_method = hi_wb_df[hi_wb_df['Method'] == method]
        lo_wbdf_method = lo_wb_df[lo_wb_df['Method'] == method]
        wbdf_method = wb_df[wb_df['Method'] == method]
        
        clean_acc = wbdf_method[wbdf_method['Perturbation Type']=='clean']['Accuracy'].mean()
        wbdf_method = wbdf_method[wbdf_method['Perturbation Type'] !='clean']
        
        
        hsev_cc_df_method = hisev_filtered_cc_df[hisev_filtered_cc_df['Method'] == method]
        lsev_cc_df_method = losev_filtered_cc_df[losev_filtered_cc_df['Method'] == method]
        cc_df_method = filtered_cc_df[filtered_cc_df['Method'] == method]
        
    #     hsev_cc_df_method = hisev_cc_df[hisev_cc_df['Method'] == method]
    #     lsev_cc_df_method = losev_cc_df[losev_cc_df['Method'] == method]
    #     cc_df_method = cc_df[cc_df['Method'] == method]
        
        cc_acc = np.mean(cc_df_method['Accuracy'].values)*100
        hsev_cc_acc = np.mean(hsev_cc_df_method['Accuracy'].values)*100
        lsev_cc_acc = np.mean(lsev_cc_df_method['Accuracy'].values)*100
        
        wb_acc = np.mean(wbdf_method['Accuracy'].values)
        hsev_wb_acc = np.mean(hi_wbdf_method['Accuracy'].values)
        lsev_wb_acc = np.mean(lo_wbdf_method['Accuracy'].values)
        
        mean_ovr = sum([clean_acc, cc_acc, wb_acc])/3
        mean_lo = sum([lsev_cc_acc, lsev_wb_acc])/2
        mean_hi = sum([hsev_cc_acc, hsev_wb_acc])/2
        mean_pert = sum([wb_acc, cc_acc])/2
        
        row = {
            'Method': method,
            'Overall Mean': mean_ovr,
            'Mean Perturbed': mean_pert,
            'Mean Low': mean_lo,
            'Mean Hi': mean_hi,
            'Mean CC': cc_acc,
            'Low CC': lsev_cc_acc,
            'High CC': hsev_cc_acc,
            'Mean WB': wb_acc,
            'Low WB': lsev_wb_acc,
            'High WB': hsev_wb_acc,
            'Clean': clean_acc,
        }
        result_rows.append(row)
    results_df = pd.DataFrame(result_rows)
    print(results_df.to_latex(index=False, float_format="%.1f"))
    print(results_df.to_latex(columns=['Method', 'Overall Mean', 'Mean CC', 'Mean WB', 'Clean'], index=False, float_format="%.1f"))

    # filtered_cc_df = cc_df[(cc_df['Corruption Method'] != 'gaussian_noise') | (cc_df['Corruption Method'] != 'gaussian_blur')]
    # hisev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] > 4]
    # losev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] < 4]
    
    # # # print(df)
    # logdicts = get_logdict(wb_plot_config)
    # wb_df = create_cc_table_df(logdicts, wb_plot_config)
    # wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [0., .002, .004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5, 1.5, 2.0])]
    
    # result_rows = []
    # for method in wb_df['Method'].unique():
    #     df_method = wb_df[wb_df['Method'] == method]
    #     type_v_acc = pd.pivot_table(df_method, values='Accuracy', index='Perturbation Type')
    #     type_v_acc = type_v_acc.to_dict()['Accuracy']
    #     type_v_acc['mean'] = np.mean(list(type_v_acc.values()))
    #     type_v_acc['mean perturbed'] = np.mean([v for k, v in type_v_acc.items() if k != 'clean'])
    #     type_v_acc['method'] = method
    #     result_rows.append(type_v_acc)
    # results_df = pd.DataFrame(result_rows)
    # print(results_df.to_latex(columns=['method', 'mean', 'mean perturbed', 'corruption', 'whitebox', 'clean'], index=False, float_format="%.1f"))
    # outdir = maybe_create_dir(f'{outdir_root}/Ecoset')

def plot_imagenet_cc_results_table(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    log_root = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs'
    cc_plot_config = OrderedDict([
            ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
            ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['CC5RandAugAPGD'])),
            ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
            ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsDetNoisedeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsCCAPGD'])),
            ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseCCAPGD'])),
            ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD'])),
    ])
    wb_plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGD_25',
                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGDL2_25'])),
        ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
    ])
    all_plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['CCAPGD', 'APGD', 'APGDL2'])),
        ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['CC5RandAug', 'CC5RandAugAPGD', '5RandAug'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsCCAPGD',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
                                                                                                                'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        ('R-Warp', (f'{log_root}/imagenet_folder-0.0/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18', ['Top5FixationsDetNoisedeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsCCAPGD',
                                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGD_25',
                                                                                                                'Top5FixationsdeepgazeIII:rwarp-6.1-7.0-7.1-in1kFixationsAPGDL2_25'])),
        ('VOneBlock', (f'{log_root}/imagenet_folder-0.0/ImagenetVOneBlockCyclicLRXResNet2x18', ['DetNoiseCCAPGD', 'DetNoiseAPGD_25', 'DetNoiseAPGDL2_25'])),
        ('AT', (f'{log_root}/imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['CCAPGD', 'APGD', 'APGDL2'])),
    ])
    cc_df = load_cc_results(cc_plot_config, '/home/mshah1/workhorse3/imagenet/distorted/test_img_paths_and_labels-190K.csv')

    logdicts = get_logdict(wb_plot_config)
    wb_df = create_cc_table_df(logdicts, wb_plot_config)

    hisev_cc_df = cc_df[cc_df['Corruption Severity'] > 3]
    losev_cc_df = cc_df[cc_df['Corruption Severity'] <= 3]
    filtered_cc_df = cc_df[(cc_df['Corruption Method'] != 'gaussian_noise') & (cc_df['Corruption Method'] != 'gaussian_blur')]
    hisev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] > 3]
    losev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] <= 3]

    wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [0., .002, .004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5, 1.5, 2.0])]
    hi_wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [.004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [1.5, 2.0])]
    lo_wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [.002]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5])]

    result_rows = []
    for method in wb_df['Method'].unique():
        hi_wbdf_method = hi_wb_df[hi_wb_df['Method'] == method]
        lo_wbdf_method = lo_wb_df[lo_wb_df['Method'] == method]
        wbdf_method = wb_df[wb_df['Method'] == method]
        
        clean_acc = wbdf_method[wbdf_method['Perturbation Type']=='clean']['Accuracy'].mean()
        wbdf_method = wbdf_method[wbdf_method['Perturbation Type'] !='clean']
        
        
        hsev_cc_df_method = hisev_filtered_cc_df[hisev_filtered_cc_df['Method'] == method]
        lsev_cc_df_method = losev_filtered_cc_df[losev_filtered_cc_df['Method'] == method]
        cc_df_method = filtered_cc_df[filtered_cc_df['Method'] == method]
        
    #     hsev_cc_df_method = hisev_cc_df[hisev_cc_df['Method'] == method]
    #     lsev_cc_df_method = losev_cc_df[losev_cc_df['Method'] == method]
    #     cc_df_method = cc_df[cc_df['Method'] == method]
        
        cc_acc = np.mean(cc_df_method['Accuracy'].values)*100
        hsev_cc_acc = np.mean(hsev_cc_df_method['Accuracy'].values)*100
        lsev_cc_acc = np.mean(lsev_cc_df_method['Accuracy'].values)*100
        
        wb_acc = np.mean(wbdf_method['Accuracy'].values)
        hsev_wb_acc = np.mean(hi_wbdf_method['Accuracy'].values)
        lsev_wb_acc = np.mean(lo_wbdf_method['Accuracy'].values)
        
        mean_ovr = sum([clean_acc, cc_acc, wb_acc])/3
        mean_lo = sum([lsev_cc_acc, lsev_wb_acc])/2
        mean_hi = sum([hsev_cc_acc, hsev_wb_acc])/2
        mean_pert = sum([wb_acc, cc_acc])/2
        
        row = {
            'Method': method,
            'Overall Mean': mean_ovr,
            'Mean Perturbed': mean_pert,
            'Mean Low': mean_lo,
            'Mean Hi': mean_hi,
            'Mean CC': cc_acc,
            'Low CC': lsev_cc_acc,
            'High CC': hsev_cc_acc,
            'Mean WB': wb_acc,
            'Low WB': lsev_wb_acc,
            'High WB': hsev_wb_acc,
            'Clean': clean_acc,
        }
        result_rows.append(row)
    results_df = pd.DataFrame(result_rows)
    print(results_df.to_latex(index=False, float_format="%.1f"))
    print(results_df.to_latex(columns=['Method', 'Overall Mean', 'Mean CC', 'Mean WB', 'Clean'], index=False, float_format="%.1f"))

def plot_ecoset10_pgdinf_ablation_results():
    plot_config = OrderedDict([
        ('Everything', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (u'\u2718' + ' Dynamic\nFixation', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationScale=3DetNoiseAPGD_25'])),
        (u'\u2718' + ' Desaturation', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500OnlyColorWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (u'\u2718' + ' Multiple\nFixations', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top1FixationScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (u'\u2718' + ' Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500NoBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        # ('No VDT', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['5FixationAPGD'])),
        (u'\u2718' + ' Noise', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3deepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        # ('Non-Adaptive-Blur with Noise', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        # ('GBlur', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        # ('Non-Adaptive Desaturation', (f'{log_root}/ecoset10-0.0/Ecoset10GreyscaleCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['DetNoiseAPGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    df = df[dataframe_or(df, 'Perturbation Distance ‖ϵ‖∞', [0., .008])]
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    print(df)
    df.to_csv(f'{outdir}/ecoset10_pgdinf_ablation.csv')
    plt.figure(figsize=(8,6))
    with sns.plotting_context("paper", font_scale=2.8):
        ax = sns.barplot(y='Method', x='Accuracy', hue='Is Perturbed', data=df, order=plot_config)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        ax.set_ylabel(None)
    # for item in ax.get_xticklabels():
    #     item.set_rotation(45)
    plt.legend([],[], frameon=False)
    # plt.legend(loc='best')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    plt.tight_layout()
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    plt.savefig(os.path.join(outdir, 'ablation_acc_bar_linf.pdf'))
    plt.close()

def plot_ecoset10_pgdinf_ablation_results_2():
    plot_config = OrderedDict([
        ('Everything', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        ('Non-Adaptive\nBlur+Noise', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('Noise', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianNoiseS2500CyclicLRRandAugmentXResNet2x18', ['DetNoiseAPGD'])),
        ('Non-Adaptive\nBlur', (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('Non-Adaptive\nDesaturation', (f'{log_root}/ecoset10-0.0/Ecoset10GreyscaleCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    df = df[dataframe_or(df, 'Perturbation Distance ‖ϵ‖∞', [0., .008])]
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    print(df)
    df.to_csv(f'{outdir}/ecoset10_pgdinf_ablation_2.csv')
    plt.figure(figsize=(8,6))
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        ax = sns.barplot(y='Method', x='Accuracy', hue='Is Perturbed', data=df, order=plot_config)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        ax.set_ylabel(None)
    # for item in ax.get_xticklabels():
    #     item.set_rotation(45)
    plt.legend([],[], frameon=False)
    # plt.legend(loc='best')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    plt.tight_layout()
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    plt.savefig(os.path.join(outdir, 'ablation_acc_bar_linf_2.pdf'))
    plt.close()

def plot_cifar_cc_results_table(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    log_root = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs'
    cc_plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['CCAPGD'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationCCAPGD'])),
    ])
    wb_plot_config = OrderedDict([
        ('ResNet', (f'{log_root}/cifar10-0.0/Cifar10CyclicLRAutoAugmentWideResNet4x22', ['APGD', 'APGDL2'])),
        ('R-Blur', (f'{log_root}/cifar10-0.0/Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22', ['5FixationAPGD', '5FixationAPGDL2'])),
    ])
    cc_df = load_cc_results(cc_plot_config, '/home/mshah1/workhorse3/cifar-10-batches-py/distorted/test_img_ids_and_labels.csv')

    logdicts = get_logdict(wb_plot_config)
    wb_df = create_cc_table_df(logdicts, wb_plot_config)

    hisev_cc_df = cc_df[cc_df['Corruption Severity'] > 3]
    losev_cc_df = cc_df[cc_df['Corruption Severity'] <= 3]
    filtered_cc_df = cc_df[(cc_df['Corruption Method'] != 'gaussian_noise') & (cc_df['Corruption Method'] != 'gaussian_blur')]
    hisev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] > 3]
    losev_filtered_cc_df = filtered_cc_df[filtered_cc_df['Corruption Severity'] <= 3]

    wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [0., .002, .004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5, 1.5, 2.0])]
    hi_wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [.004, .008]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [1.5, 2.0])]
    lo_wb_df = wb_df[dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖∞', [.002]) | dataframe_or(wb_df, 'Perturbation Distance ‖ϵ‖2', [.5])]

    result_rows = []
    for method in wb_df['Method'].unique():
        hi_wbdf_method = hi_wb_df[hi_wb_df['Method'] == method]
        lo_wbdf_method = lo_wb_df[lo_wb_df['Method'] == method]
        wbdf_method = wb_df[wb_df['Method'] == method]
        
        clean_acc = wbdf_method[wbdf_method['Perturbation Type']=='clean']['Accuracy'].mean()
        wbdf_method = wbdf_method[wbdf_method['Perturbation Type'] !='clean']
        
        
        hsev_cc_df_method = hisev_filtered_cc_df[hisev_filtered_cc_df['Method'] == method]
        lsev_cc_df_method = losev_filtered_cc_df[losev_filtered_cc_df['Method'] == method]
        cc_df_method = filtered_cc_df[filtered_cc_df['Method'] == method]
        
    #     hsev_cc_df_method = hisev_cc_df[hisev_cc_df['Method'] == method]
    #     lsev_cc_df_method = losev_cc_df[losev_cc_df['Method'] == method]
    #     cc_df_method = cc_df[cc_df['Method'] == method]
        
        cc_acc = np.mean(cc_df_method['Accuracy'].values)*100
        hsev_cc_acc = np.mean(hsev_cc_df_method['Accuracy'].values)*100
        lsev_cc_acc = np.mean(lsev_cc_df_method['Accuracy'].values)*100
        
        wb_acc = np.mean(wbdf_method['Accuracy'].values)
        hsev_wb_acc = np.mean(hi_wbdf_method['Accuracy'].values)
        lsev_wb_acc = np.mean(lo_wbdf_method['Accuracy'].values)
        
        mean_ovr = sum([clean_acc, cc_acc, wb_acc])/3
        mean_lo = sum([lsev_cc_acc, lsev_wb_acc])/2
        mean_hi = sum([hsev_cc_acc, hsev_wb_acc])/2
        mean_pert = sum([wb_acc, cc_acc])/2
        
        row = {
            'Method': method,
            'Overall Mean': mean_ovr,
            'Mean Perturbed': mean_pert,
            'Mean Low': mean_lo,
            'Mean Hi': mean_hi,
            'Mean CC': cc_acc,
            'Low CC': lsev_cc_acc,
            'High CC': hsev_cc_acc,
            'Mean WB': wb_acc,
            'Low WB': lsev_wb_acc,
            'High WB': hsev_wb_acc,
            'Clean': clean_acc,
        }
        result_rows.append(row)
    results_df = pd.DataFrame(result_rows)
    print(results_df.to_latex(index=False, float_format="%.1f"))
    print(results_df.to_latex(columns=['Method', 'Overall Mean', 'Mean Low', 'Mean Hi', 'Mean CC', 'Mean WB', 'Clean'], index=False, float_format="%.1f"))

def plot_all_neurips_many_fixation_results():
    plot_config = OrderedDict([
        ('Ecoset-10', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18',[])),
        ('Ecoset', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [''])),
        ('Imagenet', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [''])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_many_fixation_data_df(logdicts, plot_config)
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0.0) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.004)]
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Dataset'
    df = df.rename(columns=colnames, errors="raise")
    df['Is Perturbed'] = ['perturbed' if eps == 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    df = df[(df['Number of Fixation Points'] == 49) & (~df['is_worstcase'])]
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
    plt.savefig(os.path.join(outdir, 'test_acc_bar_neurips_many_fixations.png'), bbox_inches='tight')
    plt.close()

def plot_all_neurips_five_fixation_results():
    plot_config = OrderedDict([
        ('Ecoset-10', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18',['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        ('Ecoset', (f'{log_root}/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        ('Imagenet', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Dataset'
    df = df.rename(columns=colnames, errors="raise")
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.004)]
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
    plt.savefig(os.path.join(outdir, 'test_acc_bar_neurips_five_fixation.png'), bbox_inches='tight')
    plt.close()

def plot_all_neurips_AT_results():
    plot_config = OrderedDict([
        ('Ecoset-10', (f'{log_root}/ecoset10-0.008/Ecoset10AdvTrainCyclicLRRandAugmentXResNet2x18',['APGD'])),
        ('Ecoset', (f'{log_root}/ecoset-0.008/EcosetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ('Imagenet', (f'{log_root}//imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Dataset'
    df = df.rename(columns=colnames, errors="raise")
    df['Is Perturbed'] = ['perturbed' if eps > 0 else 'clean' for eps in df['Perturbation Distance ‖ϵ‖∞'].values]
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.004)]
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
    plt.savefig(os.path.join(outdir, 'test_neurips_acc_bar_AT.png'), bbox_inches='tight')
    plt.close()

def plot_imagenet_apgd_step_results():
    plot_config = OrderedDict([
        (1, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_1'])),
        (5, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_5'])),
        (10, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_10'])),
        (25, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (50, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_50'])),
        (100, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = '# Steps'
    df = df.rename(columns=colnames, errors="raise")
    df = df[df['Perturbation Distance ‖ϵ‖∞'] == 0.004]
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.2, rc={'lines.linewidth': 2.}):
        sns.lineplot(x='# Steps', y='Accuracy', markers=True, palette=cmap, data=df)
    # plt.legend(loc='best', ncol=3)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    # plt.ylim((0,1))
    plt.xticks(list(plot_config.keys()), list(plot_config.keys()))
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_steps.pdf'), bbox_inches='tight')
    plt.close()

def plot_imagenet_apgd_method_results():
    plot_config = OrderedDict([
        ('base', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        ('EOT-10', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25_EOT10'])),
        ('STE', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsStraightThroughScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    sns.set_style("whitegrid")
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0.004)]
    print(df)
    with sns.plotting_context("paper", font_scale=2.7, rc={'lines.linewidth': 2.}):
        ax = sns.barplot(x='Method', y='Accuracy', hue_order=plot_config, data=df[(df['Method'] != 'base')])
        ax.axhline(df[(df['Method'] == 'base')]['Accuracy'].mean(), linestyle='--')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge')
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_atk_methods.pdf'), bbox_inches='tight')
    plt.close()

def plot_imagenet_apgd_vscale_results():
    plot_config = OrderedDict([
        (15, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=1DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (31, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=2DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (48, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (64, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=4DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (87, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=5DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = 'Width of In-Focus Region'
    df = df.rename(columns=colnames, errors="raise")
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0.004) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]
    df['is_perturbed'] = df['Perturbation Distance ‖ϵ‖∞'] > 0.
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.2, rc={'lines.linewidth': 2.}):
        sns.lineplot(x='Width of In-Focus Region', y='Accuracy', hue='is_perturbed', markers=True, data=df)
    # plt.legend(loc='best', ncol=3)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    # plt.ylim((0,1))
    plt.xticks(list(plot_config.keys()), list(plot_config.keys()))
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_vscale.pdf'), bbox_inches='tight')
    plt.close()

def plot_imagenet_apgd_nfixations_results():
    plot_config = OrderedDict([
        (1, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top1FixationScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (2, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top2FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (3, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top3FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (4, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top4FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (5, (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = '# Fixation Points'
    df = df.rename(columns=colnames, errors="raise")
    df = df[(df['Perturbation Distance ‖ϵ‖∞'] == 0.004) | (df['Perturbation Distance ‖ϵ‖∞'] == 0.)]
    df['is_perturbed'] = df['Perturbation Distance ‖ϵ‖∞'] > 0.
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.2, rc={'lines.linewidth': 2.}):
        sns.lineplot(x='# Fixation Points', y='Accuracy', hue='is_perturbed', markers=True, data=df)
    # plt.legend(loc='best', ncol=3)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    # plt.ylim((0,1))
    plt.xticks(list(plot_config.keys()), list(plot_config.keys()))
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_nfix.pdf'), bbox_inches='tight')
    plt.close()

def plot_imagenet_autoattack_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    plot_config = OrderedDict([
        ('APGD', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationScale=3DetNoiseAPGD'])),
        ('AutoAttack', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationScale=3DetNoiseAutoAttackLinf'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    ax = _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend, figsize=(6,4))
    ax.get_legend().set_title(None)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_autoattack_linf{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_autoattack_pgdl2_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        ('APGD', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationScale=3DetNoiseAPGDL2','5FixationScale=3DetNoiseAPGD'])),
        ('AutoAttack', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['5FixationScale=3DetNoiseAutoAttackL2','5FixationScale=3DetNoiseAutoAttackLinf'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    print(df)
    ax = _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend, figsize=(6,4))
    ax.get_legend().set_title(None)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_autoattack_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_gaussian_baseline_pgdinf_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖∞'

    plot_config = OrderedDict([
        # ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD'])),
        # ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGD'])),
        ('G-Blur($\sigma=10.5$)', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianBlurCyclicLRRandAugmentXResNet2x18', ['APGD_25'])),
        ('G-Noise($\sigma=0.125$)', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['DetNoiseAPGD_25'])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        # ('AT', (f'{log_root}//imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    _plot_baseline_pgd_results(df, plot_config, norm='∞', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend, figsize=(6,4))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.24), ncol=2)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_gaussian_baseline_linf{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_imagenet_gaussian_baseline_pgdl2_results(stacked=False, min_eps=0., max_eps=1., legend=False):
    xlabel = 'Perturbation Distance ‖ϵ‖2'

    plot_config = OrderedDict([
        # ('ResNet', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
        # ('5-RandAffine', (f'{log_root}/imagenet_folder-0.0/ImagenetCyclicLRRandAugmentXResNet2x18', ['5RandAug', '5RandAugAPGDL2'])),
        ('G-Blur($\sigma=10.5$)', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianBlurCyclicLRRandAugmentXResNet2x18', ['APGD_25','APGDL2_25',])),
        ('G-Noise($\sigma=0.125$)', (f'{log_root}/imagenet_folder-0.0/ImagenetGaussianNoiseCyclicLRRandAugmentXResNet2x18', ['DetNoiseAPGD_25','DetNoiseAPGDL2_25',])),
        ('R-Blur', (f'{log_root}/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18', [
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25',
            'Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGDL2_25'])),
        # ('AT', (f'{log_root}//imagenet_folder-0.008/imagenet_folder-0.008/ImagenetAdvTrainCyclicLRRandAugmentXResNet2x18', ['APGD', 'APGDL2'])),
    ])
    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    df = df[df['Perturbation Distance ‖ϵ‖2'] != 1.5]
    outdir = maybe_create_dir(f'{outdir_root}/Imagenet')
    _plot_baseline_pgd_results(df, plot_config, norm='2', stacked=stacked, min_eps=min_eps, max_eps=max_eps, legend=legend, figsize=(6,4))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.23), ncol=3)
    plt.savefig(os.path.join(outdir, f'test_acc_bar_gaussian_baseline_l2{"_stacked" if stacked else ""}.pdf'), format='pdf')
    plt.close()

def plot_ecoset10_apgd_many_blurs_results():
    plot_config = OrderedDict([
        # (2, (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurS2CyclicLR1e_1RandAugmentXResNet2x18', ['APGD_25'])),
        ("$\sigma=0$", (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
        ("$\sigma=3$", (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurS3CyclicLR1e_1RandAugmentXResNet2x18', ['APGD_25'])),
        ("$\sigma=5$", (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurS5CyclicLR1e_1RandAugmentXResNet2x18', ['APGD_25'])),
        ("$\sigma=8$", (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurS8CyclicLR1e_1RandAugmentXResNet2x18', ['APGD_25'])),
        ("$\sigma=10.5^*$", (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        ('Ada. Blur', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurOnlyColorWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationScale=3APGD_25'])),
        # ('R-Blur\n(No Noise)', (f'{log_root}/ecoset10-0.0/Ecoset10RetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['5FixationScale=3APGD_25'])),
        # ('$\sigma_{blur}=10.5$\n$\sigma_{noise}=0.25$', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18', ['APGD'])),
        # ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = '$\sigma$'
    df = df.rename(columns=colnames, errors="raise")
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    plt.figure(figsize=(8,4))
    sns.set_style("whitegrid")
    # cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.lineplot(y='Accuracy', x='Perturbation Distance ‖ϵ‖∞', hue='$\sigma$', style='$\sigma$', 
                hue_order=plot_config, markers=True, data=df[df['Perturbation Distance ‖ϵ‖∞'] <= 0.008])
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    plt.xticks([0, 0.002, 0.004, 0.006, 0.008], [0, '', 0.004, '', 0.008])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_blur_std.pdf'), bbox_inches='tight')
    print(os.path.join(outdir, 'test_acc_bar_linf_blur_std.pdf'))
    plt.close()

def plot_ecoset10_apgd_many_WDs_results():
    plot_config = OrderedDict([
        # (2, (f'{log_root}/ecoset10-0.0/Ecoset10GaussianBlurS2CyclicLR1e_1RandAugmentXResNet2x18', ['APGD_25'])),
        ('R-Blur', (f'{log_root}/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18', ['Top5FixationsScale=3DetNoisedeepgazeIII:rblur-6.1-7.0-7.1-in1kFixationsPrecomputedFmapPcFmap-APGD_25'])),
        (5e-3, (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRWD5e_3RandAugmentXResNet2x18', ['APGD_25'])),
        (1e-3, (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRWD1e_3RandAugmentXResNet2x18', ['APGD_25'])),
        (5e-4, (f'{log_root}/ecoset10-0.0/Ecoset10CyclicLRRandAugmentXResNet2x18', ['APGD'])),
    ])

    logdicts = get_logdict(plot_config)
    df = create_data_df(logdicts, plot_config)
    colnames = {n:n for n in df.columns.to_list()}
    colnames['Method'] = '$\lambda$'
    df = df.rename(columns=colnames, errors="raise")
    print(df)
    outdir = maybe_create_dir(f'{outdir_root}/Ecoset10')
    sns.set_style("whitegrid")
    # cmap = plt.cm.get_cmap('Set1')
    # sns.barplot(x='Perturbation Distance ‖ϵ‖∞', y='Accuracy', hue='Method', hue_order=plot_config, data=df)
    with sns.plotting_context("paper", font_scale=2.8, rc={'lines.linewidth': 2.}):
        sns.lineplot(y='Accuracy', x='Perturbation Distance ‖ϵ‖∞', hue='$\lambda$', style='$\lambda$', 
                hue_order=plot_config, markers=True, data=df[df['Perturbation Distance ‖ϵ‖∞'] <= 0.008])
    # plt.legend(loc='best', ncol=3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    # plt.ylim((0,1))
    # plt.yticks([i*10 for i in range(0,11,2)], [i*10 for i in range(0,11,2)])
    # plt.legend([],[], frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar_linf_weight_decay.pdf'), bbox_inches='tight')
    plt.close()

# plot_imagenet_cc_results()

# plot_imagenet_autoattack_pgdinf_results(min_eps=0.002, max_eps=0.008, legend=True)
# plot_imagenet_autoattack_pgdl2_results(min_eps=0.5, max_eps=2.0)
plot_imagenet_gaussian_baseline_pgdinf_results(min_eps=0.002, max_eps=0.008)
plot_imagenet_gaussian_baseline_pgdl2_results(min_eps=.5, max_eps=2.)

# plot_ecoset10_apgd_many_blurs_results()
# plot_ecoset10_apgd_many_WDs_results()

# plot_imagenet_apgd_nfixations_results()
# plot_imagenet_apgd_vscale_results()
# plot_imagenet_apgd_method_results()
# plot_imagenet_apgd_step_results()

# plot_all_neurips_many_fixation_results()
# plot_all_neurips_five_fixation_results()
# plot_all_neurips_AT_results()
# plot_all_ecoset_AT_results()

# plot_imagenet_cc_results()
# plot_neurips_ecoset_cc_results()

# plot_ecoset10_pgdinf_ablation_results()
# plot_ecoset10_pgdinf_ablation_results_2()
# plot_cifar_cc_results_table()
# print(plot_imagenet_cc_results_table())
# print(plot_ecoset_cc_results_table())

# plot_cifar10_baseline_pgdinf_results(stacked=False, min_eps=.002, max_eps=.008)
# plot_ecoset10_baseline_pgdinf_results(stacked=False, min_eps=.002, max_eps=.008)
# plot_ecoset100_baseline_pgdinf_results(stacked=False, min_eps=.002, max_eps=.008)
# plot_ecoset_baseline_pgdinf_results(stacked=False, min_eps=.002, max_eps=.008)
# plot_imagenet_baseline_pgdinf_results(min_eps=.002, max_eps=.008)

# plot_cifar10_baseline_pgdl2_results(stacked=False, min_eps=.125)
# plot_ecoset10_baseline_pgdl2_results(stacked=False, min_eps=.5, max_eps=2.)
# plot_ecoset100_baseline_pgdl2_results(stacked=False, min_eps=.5, max_eps=2.)
# plot_ecoset_baseline_pgdl2_results(stacked=False, min_eps=.5, max_eps=2.)
# plot_imagenet_baseline_pgdl2_results(min_eps=.5, max_eps=2.)

# plot_ecoset_biomodels_delta_pgd_results(norm='inf', min_eps=0, max_eps=.008)
# plot_ecoset_biomodels_delta_pgd_results(norm=2, min_eps=0, max_eps=2., legend=True)
# plot_imagenet_biomodels_delta_pgd_results(norm='inf', min_eps=0, max_eps=.008)
# plot_imagenet_biomodels_delta_pgd_results(norm=2, min_eps=0, max_eps=2.)

# plot_ecoset_biomodels_pgd_results(norm='inf', min_eps=.002, max_eps=.008)
# plot_ecoset_biomodels_pgd_results(norm=2, min_eps=0.5, max_eps=2.)
# plot_imagenet_biomodels_pgd_results(norm='inf', min_eps=.002, max_eps=.008)
# plot_imagenet_biomodels_pgd_results(norm=2, min_eps=0.5, max_eps=2.)

# plot_ecoset100_pgd_all_results()
# plot_ecoset_pgd_all_results()
# plot_imagenet_pgd_all_results()
# plot_imagenet_baseline_pgd_all_results()

# plot_cifar10_baseline_pgdl2_results(stacked=False)

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
# plot_ecoset100_certified_robustness_results()
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