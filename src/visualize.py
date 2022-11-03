from argparse import ArgumentParser
import argparse
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
from adversarialML.biologically_inspired_models.src.utils import _compute_area_under_curve, load_logs, get_eps_from_logdict_key, write_pickle, load_pickle
from adversarialML.biologically_inspired_models.src import models
from torchvision.transforms.functional import erase

def get_model_from_state_dict(state_dict, args, d, num_classes):
    model = get_model(args.model_config, d, num_classes)
    state_dict = {k:torch.tensor(v).to(get_device()) for k,v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

def get_channel_influence(model, X):
    loader = torch.utils.data.DataLoader(torch.FloatTensor(X), batch_size=128, shuffle=False)
    for x in tqdm(loader):
        x = x.to(get_device())
        if isinstance(model, CustomConvolutionalClassifier):
            y = super(CustomConvolutionalClassifier, model).forward(x)
            if isinstance(y, tuple):
                y = y[0]
            logits = model.classifier(y)
            if isinstance(logits, tuple):
                logits = logits[0]
            print(y.shape, logits.shape)
            
            

def get_state_hists(model, X, apply_act_to_sh=False):
    def _maybe_activate(x):
        if apply_act_to_sh:
            return model.activation(x)
        else:
            return x
    loader = torch.utils.data.DataLoader(torch.FloatTensor(X), batch_size=128, shuffle=False)
    logits = []
    state_hists = []
    for x in tqdm(loader):
        x = x.to(get_device())
        l, sh = model(x, return_state_hist=True)
        logits.append(l.cpu().detach().numpy())
        if isinstance(sh, tuple):
            sh = [_maybe_activate(sh_).cpu().detach().numpy() for sh_ in sh]
        else:
            sh = _maybe_activate(sh).cpu().detach().numpy()
        state_hists.append(sh)
    
    if isinstance(state_hists[0], list):
        state_hists = zip(*state_hists)
        state_hists = tuple(np.concatenate(x, axis=0) for x in state_hists)
    else:
        state_hists = np.concatenate(state_hists, axis=0)    
    logits = np.concatenate(logits, axis=0)
    return logits, state_hists

def get_model_and_state_hists(state_dict, args, d, num_classes, X, apply_act_to_sh=False):
    model = get_model_from_state_dict(state_dict, args, d, num_classes)
    logits, state_hists = get_state_hists(model, X, apply_act_to_sh=apply_act_to_sh)
    return logits, state_hists

def _plot_network(input_size, output_size, W):
    if input_size > 10:
        inp_wts = W[:, :input_size]
        inp_wts = inp_wts.mean(1, keepdims=True).astype(float)
        W = np.concatenate((inp_wts, W[:, input_size:]), axis=1)
        input_size = 1
    W = W.T
    W = np.concatenate((np.zeros((W.shape[0], input_size)), W), axis=1)
    G = nx.convert_matrix.from_numpy_matrix(W, parallel_edges=True, create_using=nx.MultiDiGraph)
    nodecolors = ['blue']*len(list(G.nodes()))
    nodecolors[:input_size] = ['green']*input_size
    nodecolors[input_size:input_size+output_size] = ['red']*output_size
    edgeweights = np.array([e[2]['weight'] for e in G.edges(data=True)])
    if edgeweights.size == 0:
        edgeweights[0]
    cmap = plt.cm.get_cmap('RdBu')
    # edgecolors = (edgeweights - edgeweights.min()) / edgeweights.max()
    nx.draw_circular(G, with_labels=True, arrows=True, node_size=100, node_color=nodecolors, edge_color=edgeweights, 
                    edge_vmin=edgeweights.min(), edge_vmax=edgeweights.max(), edge_cmap=cmap, connectionstyle="arc3,rad=0.1")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = edgeweights.min(), vmax=edgeweights.max()))
    sm._A = []
    plt.colorbar(sm)

def _plot_accuracy(accs, eps, ylim=None):
    width = 0.5
    x = range(len(accs))
    plt.plot(eps, accs)
    if ylim is not None:
        plt.ylim(ylim)
    # plt.xticks(x, [e for e in eps])

def _plot_logit_hist(logit_hist, method='bars', **kwargs):
    if method == 'bars':
        width = 1/(len(logit_hist) + 1)
        for i, state in enumerate(logit_hist):
            x = np.arange(1, len(state)+1) + (i * width)
            plt.bar(x, state, width=width, align='center', label=f'stp{i}', **kwargs)
            plt.xticks(np.arange(1, len(state)+1), np.arange(len(state)))
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
        #   ncol=2, fancybox=True, shadow=True)

def _project_data(data, projection='pca', ncomponents=2, T=None):
    data_shape = data.shape
    data = data.reshape(-1, data.shape[-1])
    if T is None:
        if projection == 'pca':
            T = PCA(n_components=ncomponents)
        if projection == 'tsne':
            T = TSNE(n_components=ncomponents, n_jobs=-1)
        if projection == 'mds':
            T = MDS(n_components=2, n_jobs=-1, n_init=1, metric=False)
        new_data = T.fit_transform(data)
    else:
        new_data = T.transform(data)
    new_data = new_data.reshape(*(data_shape[:-1]),ncomponents)
    return new_data, T

def _plot_state_hist_scatter(state_hist, y, **kwargs):
    d = []
    for i in range(state_hist.shape[0]):
        for j in range(min(state_hist.shape[1], 7)):
            r = {
                    'x0':state_hist[i,j,0], 
                    'x1':state_hist[i,j,1],
                    'label':y[i],
                    'step#':j
                }
            d.append(r)
    df = pd.DataFrame(d)
    ax = sns.scatterplot(x='x0', y='x1', hue='label', style='step#', palette='pastel', data=df, **kwargs)
    return ax

def plot_models_and_accuracy(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    models = logdict['models']
    plt.figure(figsize=(18, 25))
    num_plots_per_eps = 5
    for i,(test_eps, accs) in enumerate(test_acc.items()):
        sorted_idx = sorted(range(len(accs)), key=lambda i: accs[i], reverse=True)
        selected_accs = [accs[si] for si in sorted_idx[:num_plots_per_eps]]
        vmin, vmax = min(selected_accs), max(selected_accs)
        for j, si in enumerate(sorted_idx[:num_plots_per_eps]):
            plt.subplot(2*len(test_acc.keys()), num_plots_per_eps, 2*i*num_plots_per_eps + j + 1)
            ckp = models[si]()
            W = np.array(ckp['state_dict']['weight'])
            _plot_network(ckp['attributes']['input_size'], ckp['attributes']['num_classes'], W)
            plt.title(f'acc_{test_eps}={accs[si]:.3f}')

            plt.subplot(2*len(test_acc.keys()), num_plots_per_eps, 2*i*num_plots_per_eps + num_plots_per_eps + j + 1)
            sorted_eps = sorted(test_acc.keys())
            _plot_accuracy([test_acc[e][si] for e in sorted_eps], sorted_eps)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'model_plots.png'))

def plot_train_v_test_accuracy(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    all_accs = np.concatenate(list(test_acc.values()), axis=0)
    _,bins = np.histogram(all_accs, bins='auto')
    plt.figure()
    sorted_eps = sorted(test_acc.keys())
    plt.boxplot([test_acc[e] for e in sorted_eps], labels=sorted_eps, showfliers=True)
    plt.grid()
    plt.savefig(os.path.join(outdir, 'test_acc_box.png'))

def plot_train_v_test_accuracy_multimodel(logdicts, outdir):
    data = []
    for model_name, logdict in logdicts.items():
        test_acc = logdict['metrics']['test_accs']
        # best_model_idx = np.argmax(test_acc[min(test_acc.keys())])
        for eps, accs in test_acc.items():
            atkname, eps = get_eps_from_logdict_key(eps)
            # accs = [accs[best_model_idx]]
            for a in accs:
                r = {
                    'model_name': model_name,
                    'test_eps': float(eps),
                    'acc': a,
                    'attack': atkname
                }
                data.append(r)
    df = pd.DataFrame(data)
    plt.figure()
    sns.set_style("whitegrid")
    sns.relplot(x='test_eps', y='acc', hue='model_name', col='attack', data=df, kind='line')
    plt.savefig(os.path.join(outdir, 'test_acc_line.png'))

def plot_test_accuracy_bar_multimodel(logdicts, outdir):
    data = []
    for model_name, logdict in logdicts.items():
        test_acc = logdict['metrics']['test_accs']
        # best_model_idx = np.argmax(test_acc[min(test_acc.keys())])
        for eps, accs in test_acc.items():
            atkname, eps = get_eps_from_logdict_key(eps)
            # accs = [accs[best_model_idx]]
            for a in accs:
                r = {
                    'model_name': model_name,
                    'test_eps': float(eps),
                    'acc': a,
                    'attack': atkname
                }
                data.append(r)
    df = pd.DataFrame(data)
    hue_order = sorted(logdicts.keys(), key=lambda n: df[(df['test_eps'] == 0.) & (df['model_name'] == n)]['acc'].mean())
    plt.figure(figsize=(30,10))
    sns.set_style("whitegrid")
    attacks = df['attack'].unique()
    for i, atk in enumerate(attacks):
        plt.subplot(1, len(attacks), i+1)
        plt.title(f'attack={atk}')
        plt.ylim(0, 1.)
        sns.barplot(x='test_eps', y='acc', hue='model_name', hue_order=hue_order, data=df[df['attack'] == atk])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_acc_bar.png'))

def plot_acc_v_hparams(logdicts, outdir, regex_dict={'depth':'(\d)+(?=L)', 'width_factor':'\d(?=xWide)', 'dropout_p':'\d\d(?=Dropout)', 'weight_decay':'\de_\d(?=WD)'}):
    data = []
    for model_name, logdict in logdicts.items():
        hparams = {k: float(re.search(rgx, model_name).group().replace('_','-')) for k, rgx in regex_dict.items()}
        hparams['dropout_p'] /= 10
        test_acc = logdict['metrics']['test_accs']
        # best_model_idx = np.argmax(test_acc[min(test_acc.keys())])
        for eps, accs in test_acc.items():
            atkname, eps = get_eps_from_logdict_key(eps)
            for a in accs:
                r = {
                    'model_name': model_name,
                    'test_eps': float(eps),
                    'acc': a,
                    'attack': atkname
                }
                r.update(hparams)
                data.append(r)
    df = pd.DataFrame(data)
    plt.figure(figsize=(20,30))
    test_eps = df['test_eps'].unique()
    sns.set_style("whitegrid")
    nrows = len(test_eps)
    ncols = len(regex_dict)
    for i, eps in enumerate(test_eps):
        for j, hp in enumerate(regex_dict.keys()):
            plt.subplot(nrows, ncols, i*ncols + j + 1)
            if i == 0:
                plt.title(f'eps={eps}')
            sns.boxplot(x=hp, y='acc', data=df[df['test_eps'] == eps])
    plt.savefig(os.path.join(outdir, 'test_acc_v_hp.png'))

def plot_cw_norm_multimodel(logdicts, outdir):
    data = []
    for model_name, logdict in logdicts.items():
        test_acc = logdict['metrics']['test_accs']
        data_and_preds = logdict['adv_data_and_preds']
        # best_model_idx = np.argmax(test_acc[min(test_acc.keys())])
        for lz_model_dp in data_and_preds:
            model_dp = lz_model_dp()
            x = None
            for atkstr in model_dp:
                if 'CWL2' in atkstr:
                    dp = model_dp[atkstr]
                    atkname, eps = get_eps_from_logdict_key(atkstr)
                    if eps == 0.:
                        x = dp['X']
                    else:
                        xadv = dp['X']
                        norm = np.linalg.norm((x - xadv).reshape(x.shape[0], -1), axis=1)
                        # print(x.shape, xadv.shape, atkstr, norm)
                        for n in norm:
                            r = {
                                'model_name': model_name,
                                'confidence': float(eps),
                                'attack': atkname,
                                'L2-norm': n,
                            }
                            data.append(r)
    df = pd.DataFrame(data)
    sns.set_style("whitegrid")
    attacks = df['attack'].unique()
    plt.figure(figsize=(10, 8))
    g = sns.barplot(x='confidence', y='L2-norm', hue='model_name', data=df)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'test_cw_norm.png'))

def plot_test_accuracy_auc_multimodel(logdicts, outdir, plot_confint=False):
    data = []
    for model_name, logdict in logdicts.items():
        test_acc = logdict['metrics']['test_accs']
        # test_eps = np.array([get_eps_from_logdict_key(k) for k in test_acc.keys()])
        atk2eps = {}
        for k in test_acc.keys():
            atkname, eps = get_eps_from_logdict_key(k)
            atk2eps.setdefault(atkname, []).append(eps)
        for atkname, test_eps in atk2eps.items():
            # accs = np.array(list(test_acc.values())).T
            accs = np.array([test_acc[f"{atkname}{'' if atkname == '' else '-'}{eps}"] for eps in test_eps]).T
            areas = np.array([_compute_area_under_curve(test_eps, a) for a in accs])            
            for a in areas:
                r = {
                    'attack': atkname,
                    'model_name': model_name,
                    'auc': a
                }
                data.append(r)
    df = pd.DataFrame(data)
    if plot_confint:
        plt_fn = sns.barplot
        plt_type = 'bar'
    else:
        plt_fn = sns.boxplot
        plt_type = 'box'

    plt.figure(figsize=(15,len(logdicts)))
    g = plt_fn(x='auc', y='attack', hue='model_name', data=df)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=1)
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(outdir, f'test_acc_auc_{plt_type}.png'))

def plot_test_accuracy_auc_bar_multimodel(logdicts, outdir):
    return plot_test_accuracy_auc_multimodel(logdicts, outdir, plot_confint=True)

def plot_test_accuracy_auc_box_multimodel(logdicts, outdir):
    return plot_test_accuracy_auc_multimodel(logdicts, outdir, plot_confint=False)

def plot_certified_accuracy(logdict, outdir):
    radius_step = 0.01
    data = []
    for model_data in logdict['rs_preds_and_radii']:
        model_data = model_data()
        y = np.array(model_data['Y'])
        pnr_for_sigma = model_data['preds_and_radii']
        for sigma, pnr in pnr_for_sigma.items():
            if sigma > 0.125:
                continue
            preds = np.array(pnr['Y_pred'])
            radii = np.array(pnr['radii'])

            correct = (preds == y)
            # unique_radii = np.unique(radii)
            # if unique_radii[0] > 0:
            #     unique_radii = np.insert(unique_radii, 0, 0.)
            unique_radii = np.arange(0, radii.max() + radius_step, radius_step)
            
            acc_at_radius = [(correct & (radii >= r)).mean() for r in unique_radii]

            for rad, acc in zip(unique_radii, acc_at_radius):
                r = {
                    'sigma': sigma,
                    'radius': rad,
                    'accuracy': acc
                }
                data.append(r)
    df = pd.DataFrame(data)
    plt.figure()
    sns.set_style("whitegrid")
    sns.relplot(x='radius', y='accuracy', col='sigma', data=df, kind='line')
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

def plot_certified_accuracy_multimodel(logdicts, outdir):
    radius_step = 0.01
    data = []
    for model_name, logdict in logdicts.items():
        for model_data in logdict['rs_preds_and_radii']:
            model_data = model_data()
            y = np.array(model_data['Y'])[:100]
            pnr_for_sigma = model_data['preds_and_radii']
            for sigma, pnr in pnr_for_sigma.items():
                if isinstance(sigma, str):
                    exp_name, s = get_eps_from_logdict_key(sigma)
                    if len(exp_name) > 0:
                        exp_name = '-'+exp_name
                else:
                    continue
                    # s = sigma
                    # exp_name = ''
                # if s > 0.125:
                #     continue
                if 'Y' in pnr:
                    y = np.array(pnr['Y'])
                preds = np.array(pnr['Y_pred'])[:100]
                radii = np.array(pnr['radii'])[:100]
                print(model_name, sigma, preds.shape, radii.shape, y.shape)
                correct = (preds == y[: len(preds)])
                # unique_radii = np.unique(radii)
                # if unique_radii[0] > 0:
                #     unique_radii = np.insert(unique_radii, 0, 0.)
                unique_radii = np.arange(0, radii.max() + radius_step, radius_step)
                
                acc_at_radius = [(correct & (radii >= r)).mean() for r in unique_radii]

                for rad, acc in zip(unique_radii, acc_at_radius):
                    r = {
                        'sigma': s,
                        'model_name': f'{model_name}{exp_name}',
                        'radius': rad,
                        'accuracy': acc
                    }
                    data.append(r)
    df = pd.DataFrame(data)
    plt.figure()
    sns.set_style("whitegrid")
    sns.relplot(x='radius', y='accuracy', hue='model_name', col='sigma', data=df, kind='line')
    plt.savefig(os.path.join(outdir, 'rs_certified_acc_line.png'))

def plot_test_sparsity_multimodel(logdicts, outdir):
    data = []
    for model_name, logdict in logdicts.items():
        test_acc = logdict['metrics']['test_accs']
        test_acc = {float(k):v for k,v in test_acc.items()}
        best_model_idx = get_best_model_idx(test_acc)
        sparsity = max([d.get('angular_sparsity', 0.) for d in logdict['model_metrics']])
        r = {
            'model_name': model_name,
            'sparsity': sparsity
        }
        data.append(r)
    df = pd.DataFrame(data)
    print(df)
    plt_fn = sns.barplot
    plt_type = 'bar'

    plt.figure(figsize=(15,len(logdicts)))
    plt_fn(x='sparsity', y='model_name', data=df)
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(outdir, f'test_sparsity_auc_{plt_type}.png'))

def get_reconstruction(model: torch.nn.Module, x, niters=10000, lr=1.):
    orig_act = model(x).detach()
    print(orig_act.shape)
    orig_x = x

    x = torch.empty_like(orig_x).uniform_(0.45,0.55)
    x = x.requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9, nesterov=True)
    for i in range(niters):
        act = model(x)
        # loss = 0.5*((act - orig_act)**2).sum() / x.shape[0]
        loss = torch.div(torch.norm(act - orig_act, dim=1), torch.norm(orig_act, dim=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 1000) == 0:
            print(i, float(loss), float(torch.norm(x.grad, p=2, dim=1).mean()))
    return x

def get_reconstruction2(model: torch.nn.Module, x, niters=20000):
    model = model.requires_grad_(False)
    orig_act = model(x).detach()
    orig_x = x
    back_layer = torch.nn.Linear(orig_act.shape[1], x.shape[1]).to(x.device)
    print(back_layer)
    # x = torch.empty_like(orig_x).zero_()#uniform_(0.45,0.55)
    # x = x.requires_grad_(True)
    # optimizer = torch.optim.SGD([x], lr=1., momentum=0.9, nesterov=True)
    optimizer = torch.optim.SGD(back_layer.parameters(), lr=1., momentum=0.9, nesterov=True)
    for i in range(niters):
        # act = model(x)
        recon = back_layer(model(x))
        loss = 0.5*((recon - x)**2).sum() / x.shape[0]
        # loss = 0.5*((act - orig_act)**2).sum() / x.shape[0]
        # loss = torch.div(torch.norm(act - orig_act, dim=1), torch.norm(orig_act, dim=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 1000) == 0:
            print(i, float(loss))#, float(torch.norm(x.grad, p=2, dim=1).mean()))
    return recon

def plot_reconstruction(logdict, outdir, outfile_prefix='', transform=None):
    test_acc = logdict['metrics']['test_accs']
    data_and_preds = logdict['adv_data_and_preds']
    print(list(data_and_preds[0]().keys()))
    model_paths = logdict['model_paths']
    num_plots_per_eps = 10
    print(test_acc.keys())
    for i,(test_eps, accs) in enumerate(sorted(test_acc.items(), key=lambda x:x[0])):
        atkname, eps = get_eps_from_logdict_key(test_eps)
        if eps > 0:
            continue
        best_model_idx = np.argmax(accs)
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        model_clean_data_and_preds = data_and_preds[best_model_idx]()[min(test_acc.keys())]
        model_path = model_paths[best_model_idx]
        model: torch.nn.Module = torch.load(f'{model_path}/checkpoints/model_checkpoint.pt').cuda()
        model = model.requires_grad_(False)
        model = model.feature_model
        for m in model.modules():
            if isinstance(m, models.ConsistentActivationLayer):
                ca_layer = m 
                model = torch.nn.Sequential(model, ca_layer.activation)
                # model_ = model
                # model = lambda x: ca_layer.activation(model_(x, return_state_hist=True)[1][0][0])
                break

        y = np.array(model_data_and_preds['Y'])
        y_pred = np.array(model_data_and_preds['Y_pred'])
        y_clean_pred = np.array(model_clean_data_and_preds['Y_pred'])
        x = np.array(model_data_and_preds['X'])
        x_clean = np.array(model_clean_data_and_preds['X'])

        if 'selected_idx' not in locals():
            selected_idx = np.random.choice(len(x), num_plots_per_eps, replace=False)
        selected_x = x[selected_idx]
        selected_y = y[selected_idx]
        selected_pred = y_pred[selected_idx]
        
        selected_x = torch.FloatTensor(selected_x).cuda()
        if transform is not None:
            selected_x = transform(selected_x)
        # state_hist = ca_layer.activation(model(selected_x, return_state_hist=True)[1][0])
        # state_hist = state_hist[:, [0] + list(range(1, state_hist.shape[1], (state_hist.shape[1]-1)//4))]
        # reconstructions = ca_layer.back(state_hist).cpu().detach().numpy()
        if 'ca_layer' in locals():
            T = ca_layer.max_test_time_steps
            reconstructions = []
            for t in ([1, 4, 8, -1]):
                ca_layer.max_test_time_steps = t
                reconstructions.append(get_reconstruction(model, selected_x, 2500))
            reconstructions = torch.stack(reconstructions, 1)
        else:
            reconstructions = get_reconstruction(model, selected_x, 20_000, lr=0.8)
        if reconstructions.dim() == 2:
            reconstructions = reconstructions.unsqueeze(1)
        reconstructions = reconstructions.cpu().detach().numpy()
        reconstructions = reconstructions.reshape(reconstructions.shape[0], reconstructions.shape[1], *(x.shape[1:]))
        selected_x = selected_x.cpu().detach().numpy()
        print(reconstructions.shape)

        plt.figure(figsize=(18, 25))
        nrows = num_plots_per_eps
        ncols = reconstructions.shape[1]+1
        j = 1
        def plot_image(img, j):
            plt.subplot(nrows, ncols, j)
            img = np.transpose(img, (1,2,0))
            # if img.shape[2] == 1:
            #     img = np.repeat(img, 3, 2)
            plt.imshow(img)

        for y, p, orig, recon in zip(selected_y, selected_pred, selected_x, reconstructions):
            plot_image(orig, j)
            plt.title(f'L={y} P={p}')
            j += 1
            for rt in recon:
                plot_image(rt, j)
                j += 1
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{outfile_prefix}recon_{test_eps}.png'))

def plot_noisy_reconstruction(logdict, outdir):
    add_noise = lambda x: x + torch.empty_like(x).normal_(std=.5)
    plot_reconstruction(logdict, outdir, 'gnoise1', transform=add_noise)

def plot_occluded_reconstruction(logdict, outdir):
    def occlude(x):
        for _ in range(5):
            i,j = np.random.randint(x.shape[2]-1), np.random.randint(x.shape[3]-1)
            h, w = np.random.randint(1,x.shape[2]-i), np.random.randint(1,x.shape[3]-j)
            x = erase(x, i, j, h, w, 0.)
        return x
    plot_reconstruction(logdict, outdir, 'occluded_', transform=occlude)

def plot_activations_over_time(logdict, outdir, outfile_prefix='', transform=None):
    test_acc = logdict['metrics']['test_accs']
    data_and_preds = logdict['adv_data_and_preds']
    print(list(data_and_preds[0]().keys()))
    model_paths = logdict['model_paths']
    num_samples = 500
    steps = [1, 4, 8, 16]
    print(test_acc.keys())
    acc_data = []
    for i,(test_eps, accs) in enumerate(sorted(test_acc.items(), key=lambda x:x[0])):
        atkname, eps = get_eps_from_logdict_key(test_eps)
        if (atkname != 'APGD') and (eps not in [0, 0.05]):
            continue
        best_model_idx = np.argmax(accs)
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        model_clean_data_and_preds = data_and_preds[best_model_idx]()[min(test_acc.keys())]
        model_path = model_paths[best_model_idx]
        model: torch.nn.Module = torch.load(f'{model_path}/checkpoints/model_checkpoint.pt').cuda()
        model = model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, models.ConsistentActivationLayer):
                ca_layer = m
                break

        y = np.array(model_data_and_preds['Y'])
        y_pred = np.array(model_data_and_preds['Y_pred'])
        y_clean_pred = np.array(model_clean_data_and_preds['Y_pred'])
        x = np.array(model_data_and_preds['X'])
        x_clean = np.array(model_clean_data_and_preds['X'])

        if 'selected_idx' not in locals():
            selected_idx = np.random.choice(len(x), num_samples, replace=False)
        selected_x = x[selected_idx]
        selected_y = y[selected_idx]
        selected_pred = y_pred[selected_idx]
        
        selected_x = torch.FloatTensor(selected_x).cuda()
        if transform is not None:
            selected_x = transform(selected_x)
        if 'ca_layer' in locals():
            state_hist = ca_layer.activation(model(selected_x, return_state_hist=True)[1][0])
            state_hist = state_hist[:, steps]
        else:
            state_hist = model.feature_model(selected_x).unsqueeze(1)
        logits = model.classifier(state_hist).cpu().detach().numpy()
        state_hist = state_hist.cpu().detach().numpy()

        preds = np.argmax(logits, axis=-1)
        acc = (np.expand_dims(selected_y, 1) == preds).astype(float).mean(0)
        for s, a in zip(steps, accs):
            r = {'step':s, 'accuracy': a, 'perturbation size': eps}
            acc_data.append(r)

        pca = PCA(2)
        pca_act = pca.fit_transform(state_hist.reshape(-1, state_hist.shape[-1]))
        pca_act = pca_act.reshape(*(state_hist.shape[:-1]), -1)
        act_data = []
        for y, act in zip(selected_y, pca_act):
            for s,a in zip(steps, act):
                r = {
                    'x0': a[0],
                    'x1': a[1],
                    'y': y,
                    'step':s,
                }
                act_data.append(r)
        act_df = pd.DataFrame(act_data)
        ncols = pca_act.shape[1]
        plt.figure(figsize=(4*ncols, 4))
        for j,s in enumerate(steps):
            plt.subplot(1, ncols, j+1)
            sns.scatterplot(x='x0', y='x1', hue='y', palette='pastel', data=act_df[act_df['step'] == s])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'act_scatter_{test_eps}.png'))
    
    acc_df = pd.DataFrame(acc_data)
    plt.figure()
    sns.lineplot(x='step', y='accuracy', hue='perturbation size', data=acc_df)
    plt.savefig(os.path.join(outdir, 'acc_over_time.png'))

def compute_corrcoef(act):
    cov = np.cov(act, rowvar=False)
    var = np.diag(cov)
    var_mat = var.reshape(-1,1).dot(var.reshape(1,-1))
    var_mat[var_mat == 0] = 1e-8
    corr = cov / np.sqrt(var_mat)
    return corr

def plot_activations_correlations(logdicts, outdir):
    num_samples = 1000
    model_idx = 0
    neuron_idx = 0
    acts = {}
    corrs = {}

    if not os.path.exists(os.path.join(outdir, 'acts.pkl')):
        for model_name, logdict in logdicts.items():
            test_acc = logdict['metrics']['test_accs']
            data_and_preds = logdict['adv_data_and_preds']
            model_paths = logdict['model_paths']
            model: torch.nn.Module = torch.load(f'{model_paths[model_idx]}/checkpoints/model_checkpoint.pt').cuda()
            for i,(test_eps, accs) in enumerate(sorted(test_acc.items(), key=lambda x:x[0])):
                print(test_eps)
                atkname, eps = get_eps_from_logdict_key(test_eps)
                if 'APGD' not in atkname:
                    continue
                model_idx = np.argmax(accs)
                model_data_and_preds = data_and_preds[model_idx]()[test_eps]    

                x = np.array(model_data_and_preds['X'])

                if 'selected_idx' not in locals():
                    selected_idx = np.random.choice(len(x), num_samples, replace=False)
                selected_x = x[selected_idx]
                act = model.feature_model(torch.FloatTensor(selected_x).cuda()).cpu().detach().numpy()
                acts.setdefault(model_name, {})[eps] = act

        write_pickle(acts, os.path.join(outdir, 'acts.pkl'))
    else:
        acts = load_pickle(os.path.join(outdir, 'acts.pkl'))

    corrs = {}
    for model_name,  eps2act in acts.items():
        corrs[model_name] = {eps: compute_corrcoef(act) for eps, act in eps2act.items()}
        
    all_neuron_corr_data = []
    for model_name in corrs:
        clean_corr = corrs[model_name][0.]
        clean_corr_nz = np.diag(clean_corr) != 0
        for eps, corr in corrs[model_name].items():
            # diff = np.abs(clean_corr - corr / (clean_corr + 1e-8)).flatten()
            corr_nz = (np.diag(corr) != 0) & clean_corr_nz
            print(clean_corr_nz.astype(int).sum(), corr_nz.astype(int).sum())
            diff = np.abs(clean_corr[corr_nz] - corr[corr_nz]).flatten()
            print(eps, diff.min(), np.median(diff), diff.max())
            for d in diff:
                r = {
                    'Model': model_name,
                    'Perturbation Size': eps,
                    'Post-Perturbation Change in ρ': d
                }
                all_neuron_corr_data.append(r)
    df = pd.DataFrame(all_neuron_corr_data)
    df = df[(df['Perturbation Size'] == 0.1)]
    plt.figure(figsize=(5,4))
    with sns.plotting_context("paper", font_scale=2, rc={'lines.linewidth': 2}):
        sns.set_style("whitegrid")
        sns.displot(x='Post-Perturbation Change in ρ', hue='Model', data=df, kind='ecdf', legend=False, aspect=5/4, height=4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'corr_change_ecdf.png'))
            

    corr_mat_diff_data = []
    for model_name in corrs:
        clean_corr = corrs[model_name][0.]
        for eps, corr in corrs[model_name].items():
            dnorm = np.linalg.norm(clean_corr - corr, ord='fro') #/ np.linalg.norm(clean_corr, ord='fro')
            r = {
                'Model': model_name,
                'Perturbation Size': eps,
                'Post-Perturbation\nChange in R ($∥.∥_F$)': dnorm
            }
            corr_mat_diff_data.append(r)
    df = pd.DataFrame(corr_mat_diff_data)
    plt.figure(figsize=(5,4))
    with sns.plotting_context("paper", font_scale=2, rc={'lines.linewidth': 2}):
        sns.set_style("whitegrid")
        sns.lineplot(x='Perturbation Size', y='Post-Perturbation\nChange in R ($∥.∥_F$)', hue='Model', data=df)
    # plt.xticks([0., 2.5e-4, 5e-4, 1e-3], ['0.0', '2.5e-4', '5e-4', '1e-3'])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'corr_change_norm.png'))

def plot_adv_img_and_logits(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    state_hists = logdict['state_hists']
    data_and_preds = logdict['data_and_preds']
    models = logdict['models']

    plt.figure(figsize=(18, 25))
    num_plots_per_eps = 6
    ncols = num_plots_per_eps + 1
    nrows = 2*len(test_acc.keys())
    for i,(test_eps, accs) in enumerate(test_acc.items()):
        best_model_idx = np.argmax(accs)
        model_state_hist = np.array(state_hists[best_model_idx]()[str(test_eps)])
        model_clean_state_hist = np.array(state_hists[best_model_idx]()['0.0'])
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        model_clean_data_and_preds = data_and_preds[best_model_idx]()[0.0]
        ckp = models[best_model_idx]()
        input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']
        y = np.array(model_data_and_preds['Y'])
        y_pred = np.array(model_data_and_preds['Y_pred'])
        y_clean_pred = np.array(model_clean_data_and_preds['Y_pred'])
        x = np.array(model_data_and_preds['X']).squeeze()
        if len(x.shape) == 4:
            x = np.transpose(x, (0,2,3,1))
        pos_idx = np.random.choice(np.arange(len(x))[y == y_pred], size=num_plots_per_eps//2)
        neg_idx = np.random.choice(np.arange(len(x))[((y == y_clean_pred) | (test_eps == 0.)) & (y != y_pred)], size=num_plots_per_eps//2)
        selected_idx = np.concatenate((pos_idx, neg_idx), axis=0)
        plt.subplot(nrows, ncols, 2*i*ncols + 1)
        if 'weight' in ckp['state_dict']:
            W = np.array(ckp['state_dict']['weight'])
            _plot_network(input_size, num_classes, W)
        plt.title(f'acc_{test_eps}={accs[best_model_idx]:.3f}')
        for j,k in enumerate(selected_idx):
            plt.subplot(nrows, ncols, 2*i*ncols + j + 2)
            plt.imshow(x[k])
            plt.title(f'L={y[k]}, P={y_pred[k]}')
            plt.subplot(nrows, ncols, (2*i + 1)*ncols + j + 2)
            adv_logits = model_state_hist[k][:, :num_classes]
            cln_logits = model_clean_state_hist[k][:, :num_classes]
            _plot_logit_hist(softmax(adv_logits, axis=1))
            _plot_logit_hist(softmax(cln_logits, axis=1), alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'adv_img_and_logits.png'))

def maybe_convert_to_float(s):
    try:
        f = float(s)
    except:
        f = s
    return f

def plot_adv_imgs(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {maybe_convert_to_float(k):v for k,v in test_acc.items()}
    data_and_preds = logdict['adv_data_and_preds']
    print(list(data_and_preds[0]().keys()))
    models = logdict['models']

    plt.figure(figsize=(18, 25))
    num_plots_per_eps = 10
    num_positive = 3
    ncols = num_plots_per_eps
    nrows = len(test_acc.keys()) * 2
    for i,(test_eps, accs) in enumerate(sorted(test_acc.items(), key=lambda x:x[0])):
        best_model_idx = np.argmax(accs)
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        model_clean_data_and_preds = data_and_preds[best_model_idx]()[min(test_acc.keys())]
        y = np.array(model_data_and_preds['Y'])
        y_pred = np.array(model_data_and_preds['Y_pred'])
        y_clean_pred = np.array(model_clean_data_and_preds['Y_pred'])
        x = np.array(model_data_and_preds['X']).squeeze()
        x_clean = np.array(model_clean_data_and_preds['X']).squeeze()
        if len(x.shape) == 4:
            x = np.transpose(x, (0,2,3,1))
            x_clean = np.transpose(x_clean, (0,2,3,1))
        if '-' in test_eps:
            [attack_type, test_eps] = test_eps.split('-')
            test_eps = float(test_eps)
        else:
            attack_type = ''
        pos_idx = np.arange(len(x))[(y == y_pred)]
        if len(pos_idx) > 0:
            pos_idx = np.random.choice(pos_idx, size=num_positive)
        else:
            pos_idx = [-1]*(num_positive)
        neg_idx = np.arange(len(x))[((y == y_clean_pred) | (test_eps == 0.)) & (y != y_pred)]
        if len(neg_idx) > 0:
            neg_idx = np.random.choice(neg_idx, size=num_plots_per_eps-num_positive)
        else:
            neg_idx = [-1]*(num_plots_per_eps-num_positive)
        selected_idx = np.concatenate((pos_idx, neg_idx), axis=0)
        print(test_eps, selected_idx)
        for j,k in enumerate(selected_idx):
            plt.subplot(nrows, ncols, 2*i*ncols + j + 1)
            if j == 0:
                plt.title(f'acc_{attack_type}{test_eps}={accs[best_model_idx]:.3f}')
            if k != -1:
                plt.imshow(x_clean[k])
                plt.subplot(nrows, ncols, (2*i + 1)*ncols + j + 1)
                plt.imshow(x[k])
                plt.title(f'L={y[k]}, P={y_pred[k]}\nL2={np.linalg.norm((x_clean[k]-x[k]).flatten()):.2e}')
        
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'adv_imgs.png'))

def plot_state_hists(logdict, outdir, projection='pca'):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    state_hists = logdict['state_hists']
    data_and_preds = logdict['data_and_preds']
    models = logdict['models']

    ncols = len(test_acc.keys()) + 1
    for i,(test_eps, accs) in enumerate(test_acc.items()):
        best_model_idx = np.argmax(accs)
        model_state_hist_dict = state_hists[best_model_idx]()
        model_state_hist = np.array(model_state_hist_dict[str(test_eps)])
        model_clean_state_hist = np.array(model_state_hist_dict['0.0'])
        # idx_to_plot = np.random.choice(range(len(model_state_hist)), size=500)
        model_state_hist = model_state_hist
        model_clean_state_hist = model_clean_state_hist
        proj_model_state_hist, _ = _project_data(model_state_hist, projection=projection)
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        ckp = models[best_model_idx]()
        if i == 0:
            plt.figure(figsize=(25,int(2*model_state_hist.shape[1])))
        nrows = 2 + model_state_hist.shape[1] + 2
        input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']
        y = np.array(model_data_and_preds['Y'])
        
        plt.subplot(nrows, ncols, i+1)
        W = np.array(ckp['state_dict']['weight'])
        _plot_network(input_size, num_classes, W)
        plt.title(f'acc_{test_eps}={accs[best_model_idx]:.3f}')

        plt.subplot(nrows, ncols, ncols+i+1)
        _plot_state_hist_scatter(proj_model_state_hist, y, legend='brief' if i == 0 else False)
        for j in range(model_state_hist.shape[1]):
            plt.subplot(nrows, ncols, (j+2)*ncols+i+1)
            _plot_state_hist_scatter(proj_model_state_hist[:, [j]], y, legend=False)

        # state_adv_to_clean_dist = np.linalg.norm(model_state_hist - model_clean_state_hist, ord=2, axis=-1)
        # _, bins = np.histogram(state_adv_to_clean_dist.flatten(), bins='auto')
        # plt.subplot(nrows, ncols, (j+3)*ncols+i+1)
        # plt.boxplot(state_adv_to_clean_dist, labels=range(1, state_adv_to_clean_dist.shape[1]+1))
        
        class_probs = softmax(model_state_hist[:,:,:num_classes], axis=-1)
        class_probs = np.transpose(class_probs, [0, 2, 1])
        true_class_probs = class_probs[np.arange(class_probs.shape[0]), y]
        plt.subplot(nrows, ncols, (j+3)*ncols+i+1)
        plt.boxplot(true_class_probs, labels=range(1, true_class_probs.shape[1]+1))

        preds = np.argmax(class_probs, axis=1)
        _y = np.tile(y.reshape(-1,1), (1, preds.shape[1]))
        acc = (preds == _y).astype(float).mean(0)
        plt.subplot(nrows, ncols, (j+4)*ncols+i+1)
        plt.plot(np.arange(preds.shape[1]), acc)
        plt.ylabel('accuracy')
        plt.xlabel('#step')

    plt.savefig(os.path.join(outdir, f'state_hist_{projection}.png'))
def plot_confmat(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    
    # best_model_idx = 0#get_best_model_idx(test_acc)
    # model_data_and_preds = logdict['data_and_preds'][best_model_idx]()
    eps2confmat = {}
    for i, model_data_and_preds in enumerate(logdict['adv_data_and_preds']):
        model_data_and_preds = model_data_and_preds()
        for eps, dp in model_data_and_preds.items():
            atkname, eps = get_eps_from_logdict_key(eps)
            preds = dp['Y_pred']
            labels = dp['Y']
            classes = np.unique(labels)
            confmat = np.zeros((len(classes), len(classes)))
            for (p, l) in zip(preds, labels):
                confmat[l, p] += 1
            eps2confmat[eps] = eps2confmat.get(eps, 0) + confmat
    nrows = 1
    ncols = len(eps2confmat)
    plt.figure(figsize=(4*ncols, 4))
    for i, (eps, confmat) in enumerate(eps2confmat.items()):
        acc = np.diag(confmat).sum() / confmat.sum()
        confmat = confmat / (confmat.sum(1, keepdims=True))
        print(sorted(enumerate(np.diag(confmat)), key=lambda x: x[1]))
        plt.subplot(nrows, ncols, i+1)
        plt.title(f'acc_{eps}={acc:.3f}')
        sns.heatmap(confmat, annot=False, fmt='.1f', cbar=True)
        plt.xlabel("Prediction")
        plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'confmat.png'))

def plot_lateral_matrix(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    models = logdict['models']
    plt.figure(figsize=(25,6))
    nrows = 2
    ncols = len(test_acc)
    for i,(test_eps, accs) in enumerate(test_acc.items()):
        best_model_idx = np.argmax(accs)
        ckp = models[best_model_idx]()
        W = np.array(ckp['state_dict']['weight'])
        input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']

        plt.subplot(nrows, ncols, i+1)
        _plot_network(input_size, num_classes, W)
        plt.title(f'acc_{test_eps}={accs[best_model_idx]:.3f}')

        plt.subplot(nrows, ncols, ncols+i+1)
        W_lat = W[:, input_size:]
        cmap = plt.cm.get_cmap('RdBu')
        sns.heatmap(W_lat, annot=False, cbar=True, cmap=cmap, center=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'lateral_matrix.png'))

def plot_accuracy_v_steps(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_eps_list = [float(k) for k in test_acc.keys()]
    state_hists = logdict['state_hists']
    data_and_preds = logdict['data_and_preds']
    models = logdict['models']

    nrows = 1
    ncols = 2
    true_class_probs = {}
    accs = {}
    for j in range(len(models)):
        model_state_hist_dict = state_hists[j]()
        model_data_and_preds_dict = data_and_preds[j]()
        ckp = models[j]()
        for i,test_eps in enumerate(test_eps_list):
            model_data_and_preds = model_data_and_preds_dict[test_eps]
            model_state_hist = np.array(model_state_hist_dict[str(test_eps)])
            input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']
            y = np.array(model_data_and_preds['Y'])
            
            class_probs = softmax(model_state_hist[:,:,:num_classes], axis=-1)
            class_probs = np.transpose(class_probs, [0, 2, 1])
            tc_probs = class_probs[np.arange(class_probs.shape[0]), y]
            nsteps = np.tile(np.arange(tc_probs.shape[1]), tc_probs.shape[0])
            true_class_probs.setdefault('P(true_class)',[]).extend(tc_probs.reshape(-1).tolist())
            true_class_probs.setdefault('#steps',[]).extend(nsteps.reshape(-1).tolist())
            true_class_probs.setdefault('test_eps', []).extend([test_eps] * np.prod(tc_probs.shape))

            preds = np.argmax(class_probs, axis=1)
            _y = np.tile(y.reshape(-1,1), (1, preds.shape[1]))
            acc = (preds == _y).astype(float).mean(0)
            accs.setdefault('accuracy', []).extend(acc.tolist())
            accs.setdefault('#steps', []).extend(range(len(acc)))
            accs.setdefault('test_eps', []).extend([test_eps] * len(acc))
        
    df_true_class_probs = pd.DataFrame(true_class_probs)
    df_accs = pd.DataFrame(accs)

    plt.figure()
    sns.relplot(x='#steps', y='P(true_class)', hue='test_eps', data=df_true_class_probs, kind='line', legend='full')
    plt.savefig(os.path.join(outdir, 'prob_v_step.png'))
    sns.relplot(x='#steps', y='accuracy', hue='test_eps', data=df_accs, kind='line', legend='full')
    plt.savefig(os.path.join(outdir, 'acc_v_step.png'))

def get_state_hist_and_logits(sd, args, input_size, num_classes, X, apply_act_to_sh=False):
    _, model_state_hist = get_model_and_state_hists(sd, args, input_size, num_classes, X, apply_act_to_sh=apply_act_to_sh)
    if isinstance(model_state_hist, tuple):
        logits = model_state_hist[-1]
        model_state_hist = model_state_hist[-2]
    else:
        logits = model_state_hist[:, :, :num_classes]
    return model_state_hist, logits

def compute_f_ratio(X, Y):
    X = torch.from_numpy(X)
    classwise_X = [X[Y == y] for y in np.unique(Y)]
    classwise_mean_X = [cwx.mean(0, keepdims=True) for cwx in classwise_X]
    classwise_total_X = [cwx.sum(0, keepdims=True) for cwx in classwise_X]
    mean_X = X.mean(0, keepdims=True)
    total_X = X.sum(0, keepdims=True)
    t0 = time.time()
    H = torch.stack([cwx.shape[0] * (cwm - mean_X).T.mm(cwm - mean_X) for cwm, cwx in zip(classwise_mean_X, classwise_X)], 0).sum(0).numpy()
    E = torch.stack([torch.bmm((cwx - cwm).unsqueeze(2), (cwx-cwm).unsqueeze(1)).sum(0) for cwm, cwx in zip(classwise_mean_X, classwise_X)]).sum(0).numpy()
    print(time.time() - t0)
    classwise_total_X = torch.stack(classwise_total_X, 0)
    
    t0 = time.time()
    A = (torch.bmm(classwise_total_X.transpose(1,2), classwise_total_X) / torch.tensor([x.shape[0] for x in classwise_X]).reshape(-1,1,1)).sum(0)
    H2 = A - total_X.T.mm(total_X) / X.shape[0]
    H2 = H2.numpy()
    E2 = torch.bmm(X.unsqueeze(2), X.unsqueeze(1)).sum(0) - A
    E2 = E2.numpy()
    print(time.time() - t0)
    # print(H.shape, H2.shape, E.shape, E2.shape)

    # print((H == H2).all(), (E == E2).all())

    # print(np.trace(H.dot(np.linalg.inv(E))))
    # print(np.linalg.det(E) , np.linalg.det(H + E))
    
    # print(np.trace(H2.dot(np.linalg.inv(E2))))
    # print(np.linalg.det(E2) , np.linalg.det(H2 + E2))

    fscore = np.trace(H2.dot(np.linalg.inv(E2)))
    return fscore

def compute_f_ratio(X, Y):
    X = torch.from_numpy(X)
    classwise_X = [X[Y == y] for y in np.unique(Y)]
    classwise_mean_X = [cwx.mean(0, keepdims=True) for cwx in classwise_X]
    mean_X = X.mean(0)
    C = 0
    for cwx, cwm in zip(classwise_X, classwise_mean_X):
        C += torch.bmm((cwx - cwm).unsqueeze(2), (cwx - cwm).unsqueeze(1)).mean(0)
    C /= len(classwise_mean_X)
    # C = [torch.bmm((cwx - cwm).unsqueeze(2), (cwx - cwm).unsqueeze(1)) for cwx, cwm in zip(classwise_X, classwise_mean_X)]
    # C = torch.stack([c.mean(0) for c in C], 0).mean(0).numpy()
    B = torch.stack([(cwm - mean_X).T.mm(cwm - mean_X) for cwm in classwise_mean_X], 0).mean(0).numpy()
    fscore = np.trace(B.dot(np.linalg.inv(C)))
    return fscore

def plot_state_hists_gif(logdict, outdir, projection='pca', plot_acc=False):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    state_hists = logdict['state_hists']
    data_and_preds = logdict['data_and_preds']
    models = logdict['models']
    args = logdict['args']

    ncols = len(test_acc.keys()) + 1
    for i,(test_eps, accs) in enumerate(test_acc.items()):
        best_model_idx = np.argmax(accs)
        ckp = models[best_model_idx]()
        input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        model_state_hist_dict = state_hists[best_model_idx]()
        model_state_hist = np.array(model_state_hist_dict[str(test_eps)])
        if all([x is None for x in model_state_hist]):
            sd = ckp['state_dict']
            model_state_hist, logits = get_state_hist_and_logits(sd, args, input_size, num_classes, model_data_and_preds['X'])
        else:
            logits = model_state_hist[:, :, :num_classes]
        model_state_hist = model_state_hist.reshape(model_state_hist.shape[0], model_state_hist.shape[1], -1)
        # idx_to_plot = np.random.choice(range(len(model_state_hist)), size=500)
        model_state_hist = model_state_hist
        proj_model_state_hist, T = _project_data(model_state_hist, projection=projection)
        # proj_model_state_hist, _ = _project_data(model_state_hist, T=T)
        nrows = 2 + model_state_hist.shape[1] + 2
        y = np.array(model_data_and_preds['Y'])
        preds = np.argmax(logits, axis=-1)
        _y = np.tile(y.reshape(-1,1), (1, preds.shape[1]))
        is_correct = (preds == _y).astype(float)
        fscores = [compute_f_ratio(model_state_hist[:,s], y) for s in range(model_state_hist.shape[1])]
        fig = plt.figure(figsize=(15, 7))
        # scat = _plot_state_hist_scatter(proj_model_state_hist[:, [0]], y, legend=False)
        def animation_function(frame):
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.suptitle(f'Step {frame}')
            _plot_state_hist_scatter(proj_model_state_hist[:, [frame]], y, legend='full')
            plt.subplot(1, 2, 2)
            if plot_acc:
                for c in range(num_classes):
                    acc = is_correct[y == c][:, :frame].mean(0)
                    plt.plot(range(frame), acc, label=str(c))
                    plt.xticks(range(proj_model_state_hist.shape[1]))
                    plt.xlabel('#steps')
                    plt.ylabel('accuracy')
                    plt.legend()
            else:
                plt.plot(range(frame), fscores[:frame])
                plt.xticks(range(proj_model_state_hist.shape[1]))
                plt.xlabel('#steps')
                plt.ylabel('accuracy')
                plt.legend()
        anim = FuncAnimation(fig, animation_function, frames=proj_model_state_hist.shape[1], interval=25)
        anim.save(os.path.join(outdir, f'state_hist_{test_eps:.2f}_{projection}.gif'), writer='imagemagick', fps=3)

def plot_num_train_v_num_units(logdicts, outdir):
    data = []
    for model_name, logdict in logdicts.items():
        test_acc = logdict['metrics']['test_accs']
        test_eps = np.array(list(test_acc.keys()))
        accs = np.array(list(test_acc.values())).T
        areas = np.array([_compute_area_under_curve(test_eps, a) for a in accs])

        args = logdict['args']
        num_train = args.dataset_config.num_train
        num_units = args.model_config.num_units
        for eps, accs in test_acc.items():
            for ac, ar in zip(accs, areas):
                r = {
                    'test_eps': eps,
                    'acc': ac,
                    'auc': ar,
                    'num_train': num_train,
                    'num_units': num_units
                }
                data.append(r)
    df = pd.DataFrame(data)
    plt.figure(figsize=(len(test_eps)*5, 4))
    # sns.relplot(x='num_train', y='acc', hue='num_units', data=df, kind='line', col='test_eps')
    # num_train = sorted(df['num_train'].unique())
    # num_units = sorted(df['num_units'].unique())
    # mat = np.zeros((len(num_train), len(num_units)))
    nrows = 1
    ncols = len(test_eps)
    # fig, ax = plt.subplots(nrows, ncols)
    for k,eps in enumerate(test_eps):
    #     for (i,nt), (j,nu) in itertools.product(enumerate(num_train),enumerate(num_units)):
    #         mat[i, j] = df[(df['num_train'] == nt) & (df['num_units'] == nu) & (df['test_eps'] == eps)]['acc'].values.mean()*100
        plt.subplot(nrows, ncols, k+1)
        # ax[k].set_title(f'eps={eps:.3f}')
        sns.lineplot(x='num_train', y='acc', hue='num_units', data=df[df['test_eps'] == eps], legend=('full' if (k == len(test_eps)-1) else False))
    #     sns.heatmap(mat, annot=True, fmt='.1f', xticklabels=num_units, yticklabels=num_train)
    plt.savefig(os.path.join(outdir, f'test_acc_num_train_v_num_units.png'))
    
def plot_feature_maps(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    state_hists = logdict['state_hists']
    data_and_preds = logdict['data_and_preds']
    models = logdict['models']
    args = logdict['args']

    num_plots_per_eps = 4
    num_channels_per_plot = 5
    ncols = num_channels_per_plot + 1
    nrows = num_plots_per_eps * 3
    # ncols = num_plots_per_eps * num_channels_per_plot + 1
    # nrows = 2*len(test_acc.keys())
    for i,(test_eps, accs) in enumerate(test_acc.items()):
        plt.figure(figsize=(int(ncols*1.5), nrows))
        best_model_idx = np.argmax(accs)
        ckp = models[best_model_idx]()
        input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']
        model_data_and_preds = data_and_preds[best_model_idx]()[test_eps]
        model_state_hist_dict = state_hists[best_model_idx]()
        model_state_hist = np.array(model_state_hist_dict[str(test_eps)])
        x = model_data_and_preds['X'].squeeze()
        rand_idx = np.random.choice(np.arange(x.shape[0]), size=1000)
        x = x[rand_idx]
        if all([x is None for x in model_state_hist]):
            sd = ckp['state_dict']
            model_state_hist, _ = get_state_hist_and_logits(sd, args, input_size, num_classes, x, False)
        if test_eps == 0:
            clean_model_state_hist = model_state_hist
        model_clean_data_and_preds = data_and_preds[best_model_idx]()[0.0]
        y = np.array(model_data_and_preds['Y'])[rand_idx]
        y_pred = np.array(model_data_and_preds['Y_pred'])[rand_idx]
        y_clean_pred = np.array(model_clean_data_and_preds['Y_pred'])[rand_idx]
        x_clean = np.array(model_clean_data_and_preds['X'])[rand_idx]
        if len(x.shape) == 4:
            x = np.transpose(x, (0,2,3,1))
            x_clean = np.transpose(x_clean, (0,2,3,1))
        pos_idx = np.random.choice(np.arange(len(x))[y == y_pred], size=num_plots_per_eps//2)
        neg_idx = np.random.choice(np.arange(len(x))[((y == y_clean_pred) | (test_eps == 0.)) & (y != y_pred)], size=num_plots_per_eps//2)
        selected_idx = np.concatenate((pos_idx, neg_idx), axis=0)
        loc = 1
        for j, si in enumerate(selected_idx):
            x_ = x[si]
            plt.subplot(nrows, ncols, loc)
            plt.title(f'L={y[si]}, P={y_pred[si]}')
            plt.imshow(x_)
            plt.axis('off')

            plt.subplot(nrows, ncols, loc+ncols)
            plt.title(f'L={y[si]}, P={y_clean_pred[si]}')
            plt.imshow(x_clean[si])
            plt.axis('off')

            plt.subplot(nrows, ncols, loc+2*ncols)
            plt.title(f'L={y[si]}, P={y_clean_pred[si]}')
            diff = (x_ - x_clean[si]).abs()
            diff = diff.mean(-1)
            # diff *= 1/(diff.max()+1e-8)
            plt.imshow(diff)
            plt.axis('off')
            loc += 1

            sh = model_state_hist[si, -1]
            clean_sh = clean_model_state_hist[si, -1]
            ch_idx = np.random.choice(np.arange(sh.shape[0]), size=num_channels_per_plot)
            sh = sh[ch_idx]
            clean_sh = clean_sh[ch_idx]
            sh = np.stack([sh,clean_sh, sh-clean_sh], axis=0)
            sh = sh.reshape(-1, *(sh.shape[2:]))
            for k, sh_ in enumerate(sh):
                loc += int((k > 0) and (k % num_channels_per_plot == 0))
                plt.subplot(nrows, ncols, loc)
                sns.heatmap(sh_, annot=False, cbar=True, xticklabels=[], yticklabels=[])
                loc += 1
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'feature_maps_{test_eps}.png'))
        # flat_model_state_hist = model_state_hist.reshape(model_state_hist.shape[0], model_state_hist.shape[1], model_state_hist.shape[2], -1)
        
def plot_first_layer_weights(logdict, outdir):
    test_acc = logdict['metrics']['test_accs']
    test_acc = {float(k):v for k,v in test_acc.items()}
    best_model_idx = get_best_model_idx(test_acc)
    ckp = logdict['models'][best_model_idx]()
    args = logdict['args']

    sd = ckp['state_dict']
    input_size, num_classes = ckp['attributes']['input_size'], ckp['attributes']['num_classes']
    model = get_model_from_state_dict(sd, args, input_size, num_classes)
    if isinstance(model, CustomConvolutionalClassifier):
        if model.conv_model_cls == ScanningFullyConnectedNetwork2:
            W = model.conv_layers[0].fwd_layer.weight.data.detach().cpu()
    elif isinstance(model, ConvClassifier):
        W = model.conv_model[0].weight.data.detach().cpu()
    
    W_norms = np.linalg.norm(W.reshape(W.shape[0], -1), ord=2, axis=-1)
    sorted_idx = sorted(range(W.shape[0]), key=lambda i: W_norms[i], reverse=True)
    W = W[sorted_idx[:30]]
    W = W.reshape(-1, *(W.shape[2:]))
    W_unf = torch.nn.functional.unfold(W.unsqueeze(1), 3, stride=1).squeeze()
    smoothness = (W_unf - W_unf[:,[5]]).abs().mean(-1).mean(-1)
    print(f'avg_smoothness={smoothness.mean():.3f}')
    ncols = 9
    nrows = np.ceil(W.shape[0] / ncols)
    plt.figure(figsize=(int(ncols*1.5), nrows))
    for i,w in enumerate(W):
        plt.subplot(nrows, ncols, i+1)
        w_unf = W_unf[i]
        cmap = plt.cm.get_cmap(['Reds', 'Greens', 'Blues'][i % 3])
        sns.heatmap(w, cbar=True, cmap=cmap, xticklabels=[], yticklabels=[])
        plt.title(f'S={smoothness[i]:.3f}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '1st_layer_weights.png'))

def get_outdir(model_log_dir, outdir, final_dirname):
    dirs = np.array([x.split('/')[-3:] for x in model_log_dir])
    unique_dirs = [np.unique(dirs[:,i]) for i in range(dirs.shape[1])]
    concat_dirs = lambda a: '+'.join(a)
    outdir = [outdir] + [concat_dirs(d) for d in unique_dirs]
    if final_dirname is not None:
        outdir[-1] = final_dirname
    outdir = os.path.join(*outdir)
    return outdir
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_log_dir', nargs='+')
    parser.add_argument('--plot_fn', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--final_dirname', type=str)
    parser.add_argument('--labels', nargs='+', default=[])
    args = parser.parse_args()

    outdir = get_outdir(args.model_log_dir, args.outdir, args.final_dirname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(outdir)
    logdict = load_logs(args.model_log_dir, args.labels)
    plot_fn = locals()[args.plot_fn]
    plot_fn(logdict, outdir)
    # plot_models_and_accuracy(logdict, outdir)
    # plot_train_v_test_accuracy(logdict, outdir)
    # plot_adv_img_and_logits(logdict, outdir)
    # plot_state_hists(logdict, outdir, projection='tsne')
