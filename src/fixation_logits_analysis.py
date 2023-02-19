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
    # '/share/workhorse3/mshah1/ecoset-10/val',
    # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/fixation_logits/16/val',
    # '/share/workhorse3/mshah1/ecoset-100/val',
    # '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/fixation_logits/16/val',
    '/share/workhorse3/mshah1/ecoset/val',
    '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset-0.0/EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/fixation_logits/49/val/',
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()]), fixation_target='logit'
)
imgs, flogits, Y = zip(*(list(ds))) #zip(*([ds[i] for i in np.random.choice(len(ds), 10, replace=False)]))
flogits = np.stack(flogits, 0)
imgs = np.stack(imgs, 0)
Y = np.array(Y)

print(imgs.shape, flogits.shape, Y.shape)

def average_logit_pred(logits):
    return np.argmax(logits.mean(1), 1)

def vote_pred(logits):
    return np.array([x[np.argmax(np.unique(x, return_counts=True)[1])] for x in np.argmax(logits, -1)])

def is_correct(preds, labels):
    return (preds == labels)

def accuracy(logits, labels, pred_fn):
    preds = pred_fn(logits)
    is_c = is_correct(preds, labels)
    return is_c.astype(float).mean()

def oracle_accuracy(logits, labels):
    preds = np.argmax(logits, -1)
    is_c = (preds == labels.reshape(-1, 1)).any(1)
    return is_c.astype(float).mean()

def get_5point_idxs(N):
    rootN = int(np.sqrt(N))
    return [0, rootN-1, rootN * rootN//2 + rootN//2, N - rootN, N-1]

def make_features(flogits):
    X = flogits
    P = softmax(X, -1)
    H = (log_softmax(X, -1) * softmax(X, -1)).sum(-1, keepdims=True)
    X = np.concatenate([X, P, H], -1)
    return X

npts = flogits.shape[1]
five_points = get_5point_idxs(npts)
# print(five_points)
# exit()
flogits_5f = flogits[:, five_points]
print('mean logits:', accuracy(flogits_5f, Y, average_logit_pred))
print('voting:', accuracy(flogits_5f, Y, vote_pred))
print('oracle-5pt:', oracle_accuracy(flogits_5f, Y))
print(f'oracle-{npts}pt:', oracle_accuracy(flogits, Y))

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