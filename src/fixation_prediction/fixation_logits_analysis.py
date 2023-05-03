import numpy as np
from torchvision.datasets import DatasetFolder
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.special import softmax, log_softmax
from mllib.datasets.fixation_point_dataset import FixationPointDataset

import torchvision
import torch
import matplotlib.pyplot as plt

# ds = DatasetFolder(
#     # '/home/mshah1/adversarialML/biologically_inspired_models/fixation_logits/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/fixation_logits/16/val/val/',
#     # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/fixation_logits/16/val',
#     # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/fixation_logits/49/val/',
#     '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/fixation_logits/16/val',
#     lambda x: np.load(x)['fixation_logits'], extensions='npz'
# )
ds = FixationPointDataset(
    '/share/workhorse3/mshah1/ecoset-10',
    'val',
    '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/fixation_logits/49',
    # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/fixation_logits/0.004/16/',
    # '/share/workhorse3/mshah1/ecoset-100/',
    # 'val',
    # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/fixation_logits/16/',
    # '/share/workhorse3/mshah1/ecoset/',
    # 'val',
    # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/fixation_logits/49/',
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()]), fixation_target='logit'
)
print(len(ds[0]))
imgs, flogits, Y = zip(*(list(ds))) #zip(*([ds[i] for i in np.random.choice(len(ds), 10, replace=False)]))
flogits = np.stack(flogits, 0)
imgs = np.stack(imgs, 0)
Y = np.array(Y)

print(imgs.shape, flogits.shape, Y.shape)

def average_logit_pred(logits):
    return np.argmax(logits.mean(1), 1)

def top5_average_logit_pred(logits):
    return np.argsort(logits.mean(1), 1)[:, -5:]

def soft_voting_pred(logits):
    return np.argmax(softmax(logits, 2).sum(1), 1)

def top5_soft_voting_pred(logits):
    return np.argsort(softmax(logits, 2).sum(1), 1)[:, -5:]

# def vote_pred(logits):
#     return np.array([x[np.argmax(np.unique(x, return_counts=True)[1])] for x in np.argmax(logits, -1)])

def vote_pred(logits):
    # votes = np.array([np.unique(fpreds, return_counts=True)[1] for fpreds in logits.argmax(-1)])
    preds = []
    for fpreds in logits.argmax(-1):
        labels, counts = np.unique(fpreds, return_counts=True)
        pred = labels[counts.argmax()]
        preds.append(pred)
    return np.array(preds)


def top5_vote_pred(logits):
    return np.array([x[np.argmax(np.unique(x, return_counts=True)[1])] for x in np.argsort(logits, -1)[..., -5:].reshape(logits.shape[0], -1)])

def is_correct(preds, labels):
    while len(labels.shape) < len(preds.shape):
        labels = np.expand_dims(labels, -1)
    is_c = (preds == labels)
    if len(is_c.shape) == 2:
        is_c = is_c.any(1)
    return is_c

def accuracy(logits, labels, pred_fn):
    preds = pred_fn(logits)
    is_c = is_correct(preds, labels)
    return is_c.astype(float).mean()

def oracle_accuracy(logits, labels):
    preds = np.argmax(logits, -1)
    is_c = (preds == labels.reshape(-1, 1)).any(1)
    return is_c.astype(float).mean()

def all_correct_accuracy(logits, labels):
    preds = np.argmax(logits, -1)
    is_c = (preds == labels.reshape(-1, 1)).all(1)
    return is_c.astype(float).mean()

def top5_all_correct_accuracy(logits, labels):
    top5preds = np.argsort(logits, -1)[...,-5:]
    is_c = np.zeros((top5preds.shape[0],top5preds.shape[1])).astype(bool)
    for i in range(5):
        is_c |= (top5preds[...,i] == labels.reshape(-1, 1))
    is_c = is_c.all(1)
    return is_c.astype(float).mean()

def oracle_top5_accuracy(logits, labels):
    top5preds = np.argsort(logits, -1)[...,-5:]
    is_c = np.zeros((top5preds.shape[0],)).astype(bool)
    for i in range(5):
        is_c |= (top5preds[...,i] == labels.reshape(-1, 1)).any(1)
    return is_c.astype(float).mean()

def get_5point_idxs(N):
    rootN = int(np.sqrt(N))
    return [0, rootN-1, rootN * rootN//2 + rootN//2, N - rootN, N-1]

def get_topK_logits(logits, k):
    top1_probs = softmax(logits, 2).max(2)
    print(top1_probs.min(), top1_probs.mean(), np.median(top1_probs), top1_probs.max())
    topk_idxs = np.argsort(top1_probs, 1)[:, -k:]
    topk_logits = np.stack([r[tki] for tki,r in zip(topk_idxs, logits)], 0)
    return topk_logits

def get_topK_logits2(logits, k):
    logits = np.copy(logits)
    topk_logits = []
    n = int(np.sqrt(logits.shape[1]))
    for i in range(k):
        top1_probs = np.nanmax(softmax(logits, 2), 2)
        print(top1_probs[0].reshape(n, n))
        top1_idx = np.nanargmax(top1_probs, 1)
        top1_logits = logits[np.arange(logits.shape[0]), top1_idx]
        topk_logits.append(top1_logits)

        logits[np.arange(logits.shape[0]), top1_idx] += -np.inf
        offset1 = ((top1_idx % n) != (n-1)).astype(int)
        logits[np.arange(logits.shape[0]), top1_idx+offset1] += -np.inf
        offset2 = ((top1_idx % n) != 0).astype(int)
        logits[np.arange(logits.shape[0]), top1_idx-offset2] += -np.inf
        offset3 = ((top1_idx // n) > 0).astype(int)
        logits[np.arange(logits.shape[0]), top1_idx-(n*offset3)] += -np.inf
        offset4 = ((top1_idx // n) < (n-1)).astype(int)
        logits[np.arange(logits.shape[0]), top1_idx+(n*offset4)] += -np.inf
        logits[np.arange(logits.shape[0]), top1_idx+offset1-(n*offset3)] += -np.inf
        logits[np.arange(logits.shape[0]), top1_idx-offset2-(n*offset3)] += -np.inf
        logits[np.arange(logits.shape[0]), top1_idx+offset1+(n*offset4)] += -np.inf
        logits[np.arange(logits.shape[0]), top1_idx-offset2+(n*offset4)] += -np.inf
    topk_logits = np.stack(topk_logits, 1)
    return topk_logits

def get_topK_logit_combinations(logits, K):
    sum_logits = 0
    selected_logits = []
    fxpts = []
    for k in range(K):
        sum_logits = logits+sum_logits
        probs = softmax(sum_logits, 2)
        top2idx = np.argpartition(probs, logits.shape[2] - 2, 2)[:, :, -2:]
        # print(top2idx.shape, probs[0], top2idx[0])
        top2margin = np.stack([p[range(len(idxs)),idxs[:,1]]-p[range(len(idxs)),idxs[:,0]] for p,idxs in zip(probs, top2idx)],0)
        if len(fxpts) > 0:
            for pts in fxpts:
                top2margin[range(len(top2margin)), pts] *= -np.inf
        # print(top2margin.shape, probs[0], top2margin[0])
        max_margin_idxs = np.argmax(top2margin, 1)
        fxpts.append(max_margin_idxs)
        selected_logits.append(logits[range(len(logits)), max_margin_idxs])
        sum_logits = np.expand_dims(sum_logits[range(len(logits)), max_margin_idxs], 1)
    #     print(max_margin_idxs[0])
    #     print(sum_logits[0])
    #     print(logits[0, max_margin_idxs[0]])
    #     print(probs[0, max_margin_idxs[0]])
    # print([p[0] for p in fxpts])
    selected_logits = np.stack(selected_logits, 1)
    # print(selected_logits.shape)
    # print(softmax(np.cumsum(selected_logits[12], 0), 1))
    # print(sum_logits[12])
    return selected_logits

def get_topK_logit_oracle_combinations(logits, K, Y):
    sum_logits = 0
    selected_logits = []
    fxpts = []
    for k in range(K):
        sum_logits = logits+sum_logits
        probs = softmax(sum_logits, 2)
        clsprobs = np.stack([p[:, y] for p,y in zip(probs, Y)], 0)
        for p,y in zip(probs, Y):
            p[:, y] = 0
        topidx = np.argmax(probs, 2)
        # print(clsprobs[0], probs[0], topidx[0])
        # print(top2idx.shape, probs[0], top2idx[0])
        top2margin = np.stack([cp-p[range(len(idx)),idx] for cp,p,idx, y in zip(clsprobs, probs, topidx, Y)],0)
        if len(fxpts) > 0:
            for pts in fxpts:
                top2margin[range(len(top2margin)), pts] = -np.inf
        # print(top2margin.shape, probs[0], top2margin[0])
        max_margin_idxs = np.argmax(top2margin, 1)
        fxpts.append(max_margin_idxs)
        selected_logits.append(logits[range(len(logits)), max_margin_idxs])
        sum_logits = np.expand_dims(sum_logits[range(len(logits)), max_margin_idxs], 1)
        # print(max_margin_idxs[0], Y[0], top2margin[0])
        # print(sum_logits[0])
        # print(logits[0, max_margin_idxs[0]])
        # print(probs[0, max_margin_idxs[0]])
    # print([p[0] for p in fxpts])
    selected_logits = np.stack(selected_logits, 1)
    return selected_logits

def get_topK_logit_oracle_combinations2(logits, K, Y):
    from itertools import combinations
    selected_logits = []
    locidxs = []
    ncorrect = 0
    for lg,y in zip(logits,Y):
        combs = combinations(range(logits.shape[1]), K)
        for i,c in enumerate(combs):
            c = list(c)
            is_correct = np.argmax(lg[c].mean(0)) == y
            if is_correct:
                ncorrect += 1
                selected_logits.append(lg[c])
                break
        locidxs.append(i)
        if len(locidxs) > len(selected_logits):
            selected_logits.append(lg[c])
    selected_logits = np.stack(selected_logits, 0)
    acc = (selected_logits.mean(1).argmax(1) == Y).astype(float).mean()
    print(logits.shape, selected_logits.shape, ncorrect/len(logits), acc)
    print(min(locidxs), np.mean(locidxs), np.median(locidxs), max(locidxs))
    return selected_logits


def make_features(flogits):
    X = flogits
    P = softmax(X, -1)
    H = (log_softmax(X, -1) * softmax(X, -1)).sum(-1, keepdims=True)
    X = np.concatenate([X, P, H], -1)
    return X

npts = flogits.shape[1]
five_points = get_5point_idxs(npts)
flogits_top5 = get_topK_logits2(flogits, 2)
flogits_top5comb = get_topK_logit_combinations(flogits, 5)
flogits_oracletop5comb = get_topK_logit_oracle_combinations(flogits, 5, Y)
# flogits_top1comb_ex = get_topK_logit_oracle_combinations2(flogits, 1, Y)
flogits_oracletop2comb_ex = get_topK_logit_oracle_combinations2(flogits, 2, Y)
# flogits_top3comb_ex = get_topK_logit_oracle_combinations2(flogits, 3, Y)
# print(five_points)
# exit()
flogits_5f = flogits[:, five_points]
print('all points:', all_correct_accuracy(flogits, Y), top5_all_correct_accuracy(flogits, Y))
print('center:', accuracy(flogits_5f[:, [2]], Y, average_logit_pred), accuracy(flogits_5f[:, [2]], Y, top5_average_logit_pred))
print('mean logits:', accuracy(flogits, Y, average_logit_pred), accuracy(flogits, Y, top5_average_logit_pred))
print('mean logits 5pt:', accuracy(flogits_5f, Y, average_logit_pred), accuracy(flogits_5f, Y, top5_average_logit_pred))
print('mean logits top-5pt:', accuracy(flogits_top5, Y, average_logit_pred), accuracy(flogits_top5, Y, top5_average_logit_pred))
print('mean logits top-5comb:', accuracy(flogits_top5comb, Y, average_logit_pred), accuracy(flogits_top5comb, Y, top5_average_logit_pred))
print('mean logits oracle-top-5comb:', accuracy(flogits_oracletop5comb, Y, average_logit_pred), accuracy(flogits_oracletop5comb, Y, top5_average_logit_pred))
# print('mean logits oracle-top-1 exhaustive comb:', accuracy(flogits_top1comb_ex, Y, average_logit_pred), accuracy(flogits_top1comb_ex, Y, top5_average_logit_pred))
print('mean logits oracle-top-2 exhaustive comb:', accuracy(flogits_oracletop2comb_ex, Y, average_logit_pred), accuracy(flogits_oracletop2comb_ex, Y, top5_average_logit_pred))
# print('mean logits oracle-top-3 exhaustive comb:', accuracy(flogits_top3comb_ex, Y, average_logit_pred), accuracy(flogits_top3comb_ex, Y, top5_average_logit_pred))
print('voting 5pt:', accuracy(flogits_5f, Y, vote_pred), accuracy(flogits_5f, Y, top5_vote_pred))
print('voting:', accuracy(flogits, Y, vote_pred), accuracy(flogits, Y, top5_vote_pred))
print('soft voting:', accuracy(flogits, Y, soft_voting_pred), accuracy(flogits, Y, top5_soft_voting_pred))
print('soft voting 5pt:', accuracy(flogits_5f, Y, soft_voting_pred), accuracy(flogits_5f, Y, top5_soft_voting_pred))
print('soft voting top-5pt:', accuracy(flogits_top5, Y, soft_voting_pred), accuracy(flogits_top5, Y, top5_soft_voting_pred))
print('soft voting top-5comb:', accuracy(flogits_top5comb, Y, soft_voting_pred), accuracy(flogits_top5comb, Y, top5_soft_voting_pred))
print('soft voting oracle-top-5comb:', accuracy(flogits_oracletop5comb, Y, soft_voting_pred), accuracy(flogits_oracletop5comb, Y, top5_soft_voting_pred))
print('oracle-5pt:', oracle_accuracy(flogits_5f, Y), oracle_top5_accuracy(flogits_5f, Y))
print('oracle top-5comb:', oracle_accuracy(flogits_top5comb, Y), oracle_top5_accuracy(flogits_top5comb, Y))
print('oracle oracle-top-5comb:', oracle_accuracy(flogits_oracletop5comb, Y), oracle_top5_accuracy(flogits_oracletop5comb, Y))
print('oracle oracle-top-2 exhaustive comb:', oracle_accuracy(flogits_oracletop2comb_ex, Y), oracle_top5_accuracy(flogits_oracletop2comb_ex, Y))
print('oracle top-5pt:', oracle_accuracy(flogits_top5, Y), oracle_top5_accuracy(flogits_top5, Y))
print(f'oracle {npts}pt:', oracle_accuracy(flogits, Y), oracle_top5_accuracy(flogits, Y))
exit()

# def predict_fixation():
preds = np.argmax(flogits, -1)
is_c = (preds == Y.reshape(-1, 1)).astype(int)

all_c = (np.argmax(flogits, -1) == Y.reshape(-1, 1)).all(1)
any_c = (np.argmax(flogits, -1) == Y.reshape(-1, 1)).any(1)

ks = int(np.linspace(0, imgs.shape[2]-1, int(np.sqrt(npts)))[1])
imgs = torch.nn.functional.unfold(torch.from_numpy(imgs[(~all_c) & any_c]), ks, stride=ks, padding=ks//2).numpy()
imgs = np.transpose(imgs, [0, 2, 1])

X = make_features(flogits[(~all_c) & any_c])
Y = Y[(~all_c) & any_c]
print(X.shape, Y.shape)

# X_train, X_test, y_train, y_test, Y_train, Y_test, flogits_train, flogits_test = sklearn.model_selection.train_test_split(X, is_c, Y, flogits, test_size=0.25, random_state=42)
train_idx, test_idx = sklearn.model_selection.train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)

X_train = X[train_idx].reshape(-1, X.shape[-1])
y_train = is_c[train_idx].flatten()
X_test = X[test_idx].reshape(-1, X.shape[-1])
y_test = is_c[test_idx].flatten()
print(X_train.shape)

# pca2 = PCA(2)
# X_train2 = pca2.fit_transform(X_train)
# X_test2 = pca2.transform(X_test)
# print(X_train2.shape)

# X_tsne = TSNE(learning_rate='auto', init='pca', verbose=True).fit_transform((np.concatenate([X_train, X_test], 0)))
# X_tsne_train = X_tsne[:len(X_train)]
# X_tsne_test = X_tsne[len(X_train):]
# print(X_tsne.shape)

# lda = LinearDiscriminantAnalysis()
# X_lda_train = lda.fit_transform(X_train, y_train)
# X_lda_test = lda.transform(X_test)
# print(X_lda_train.shape, lda.score(X_test, y_test))

# nrows= 3
# ncols = 3
# plt.subplot(nrows,ncols,1)
# plt.scatter(X_train2[:, 0], X_train2[:, 1], c='r', alpha=0.1, label='train')
# plt.scatter(X_test2[:, 0], X_test2[:, 1], c='b', alpha=0.1, label='test')
# plt.legend()
# plt.subplot(nrows,ncols,2)
# plt.scatter(X_train2[y_train == 0][:, 0], X_train2[y_train == 0][:, 1], c='r', alpha=0.1, label='-')
# plt.scatter(X_train2[y_train == 1][:, 0], X_train2[y_train == 1][:, 1], c='b', alpha=0.1, label='+')
# plt.legend()
# plt.subplot(nrows,ncols,3)
# plt.scatter(X_test2[y_test == 0][:, 0], X_test2[y_test == 0][:, 1], c='r', alpha=0.1, label='-')
# plt.scatter(X_test2[y_test == 1][:, 0], X_test2[y_test == 1][:, 1], c='b', alpha=0.1, label='+')
# plt.legend()

# plt.subplot(nrows,ncols,ncols+1)
# plt.scatter(X_tsne_train[:, 0], X_tsne_train[:, 1], c='r', alpha=0.1, label='train')
# plt.scatter(X_tsne_test[:, 0], X_tsne_test[:, 1], c='b', alpha=0.1, label='test')
# plt.subplot(nrows,ncols,ncols+2)
# plt.scatter(X_tsne_train[y_train == 0][:, 0], X_tsne_train[y_train == 0][:, 1], c='r', alpha=0.1, label='-')
# plt.scatter(X_tsne_train[y_train == 1][:, 0], X_tsne_train[y_train == 1][:, 1], c='b', alpha=0.1, label='+')
# plt.subplot(nrows,ncols,ncols+3)
# plt.scatter(X_tsne_test[y_test == 0][:, 0], X_tsne_test[y_test == 0][:, 1], c='r', alpha=0.1, label='-')
# plt.scatter(X_tsne_test[y_test == 1][:, 0], X_tsne_test[y_test == 1][:, 1], c='b', alpha=0.1, label='+')

# plt.subplot(nrows,ncols,2*ncols+1)
# plt.scatter(X_lda_train[:, 0], np.zeros((len(X_lda_train),)), c='r', alpha=0.1, label='train')
# plt.scatter(X_lda_test[:, 0], np.zeros((len(X_lda_test),)), c='b', alpha=0.1, label='test')
# plt.subplot(nrows,ncols,2*ncols+2)
# plt.scatter(X_lda_train[y_train == 0][:, 0], np.zeros((len(X_lda_train[y_train == 0]),)), c='r', alpha=0.1, label='-')
# plt.scatter(X_lda_train[y_train == 1][:, 0], np.zeros((len(X_lda_train[y_train == 1]),)), c='b', alpha=0.1, label='+')
# plt.subplot(nrows,ncols,2*ncols+3)
# plt.scatter(X_lda_test[y_test == 0][:, 0], np.zeros((len(X_lda_test[y_test == 0]),)), c='r', alpha=0.1, label='-')
# plt.scatter(X_lda_test[y_test == 1][:, 0], np.zeros((len(X_lda_test[y_test == 1]),)), c='b', alpha=0.1, label='+')
# plt.savefig('fixation_logits_pca.png')
# exit()


# pca = PCA(.95)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# print(X_train.shape, X_test.shape)

model = LogisticRegression(class_weight='balanced', max_iter=2000, verbose=True, n_jobs=-1, solver='saga')
# model = MLPClassifier(max_iter=10_000)
# model = AdaBoostClassifier(n_estimators=100)
# model = GradientBoostingClassifier(verbose=True, n_estimators=500, n_iter_no_change=10)
# model = HistGradientBoostingClassifier(verbose=True, max_iter=5000, max_leaf_nodes=4, l2_regularization=1)
model.fit(X_train, y_train)

print('train_score:', model.score(X_train, y_train))
print('test_score:', model.score(X_test, y_test))

flogits_test = flogits[(~all_c) & any_c][test_idx]
Y_test = Y[test_idx]
flogits_test_5f = flogits_test[:, five_points]
print(flogits_test_5f.shape, Y_test.shape)
print('mean logits:', accuracy(flogits_test_5f, Y_test, average_logit_pred))
print('voting:', accuracy(flogits_test_5f, Y_test, vote_pred))

pred = model.predict_proba(X_test)[...,1]
pred = pred.reshape(-1, flogits.shape[1])
print(pred.shape, pred[:3])
pred_loc = np.argmax(pred, 1)
print(pred_loc.shape, pred_loc[:10], Y_test.shape)
logits_test = flogits_test[range(len(pred_loc)), pred_loc]
cls_pred = np.argmax(logits_test, -1)
print(flogits_test.shape, cls_pred.shape)
print((cls_pred == Y_test).astype(float).mean())

# predict_fixation()