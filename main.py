"""
SFGF: Spatial-frequency Guidance Fusion for Joint Classification of
Hyperspectral and LiDAR Data.

Select the dataset with --dataset (muufl | houston | trento); default: muufl.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as dataf
from sklearn.decomposition import PCA

import funcs

# per-dataset settings: data files, number of classes and default hyper-parameters
DATASETS = {
    'muufl': {
        'hsi_file': os.path.join('dataset', 'MUUFL', 'MUUFL.mat'),
        'hsi_key': 'HSI',
        'lidar_file': os.path.join('dataset', 'MUUFL', 'MUUFL.mat'),
        'lidar_key': 'LiDAR',
        'train_template': os.path.join('dataset', 'MUUFL', 'MUUFL_TR_20_{run}.mat'),
        'n_classes': 11,
        'patch_size': 7,
        'epoch': 600,
        'apha': 0.01,
        'gamma': 0.01,
        'lamda': 1.0,
    },
    'houston': {
        'hsi_file': os.path.join('dataset', 'Houston2013', 'Houston2013_HSI.mat'),
        'hsi_key': 'HSI',
        'lidar_file': os.path.join('dataset', 'Houston2013', 'Houston2013_DSM.mat'),
        'lidar_key': 'DSM',
        'train_template': os.path.join('dataset', 'Houston2013', 'Houston2013_TR_20_{run}.mat'),
        'n_classes': 15,
        'patch_size': 11,
        'epoch': 500,
        'apha': 0.5,
        'gamma': 0.5,
        'lamda': 1.0,
    },
    'trento': {
        'hsi_file': os.path.join('dataset', 'Trento', 'HSI.mat'),
        'hsi_key': 'HSI',
        'lidar_file': os.path.join('dataset', 'Trento', 'LiDAR.mat'),
        'lidar_key': 'LiDAR',
        'train_template': os.path.join('dataset', 'Trento', 'Trento_TR_20_{run}.mat'),
        'n_classes': 6,
        'patch_size': 11,
        'epoch': 600,
        'apha': 0.01,
        'gamma': 0.01,
        'lamda': 1.0,
    },
}

parser = argparse.ArgumentParser(description='SFGF: joint classification of hyperspectral and LiDAR data')
parser.add_argument('--dataset', type=str, default='muufl', choices=['muufl', 'houston', 'trento'],
                    help='dataset to use (default: muufl)')
parser.add_argument('--patch_size', default=None, type=int,
                    help='patch size of network input (default: dataset-specific)')
parser.add_argument('--test_freq', type=int, default=5,
                    help='evaluate the model every N epochs')
parser.add_argument('--batch_size', default=64, type=int,
                    help='training batch size')
parser.add_argument('--epoch', default=None, type=int,
                    help='number of training epochs (default: dataset-specific)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--apha', default=None, type=float,
                    help='weight of the HSI-branch loss (default: dataset-specific)')
parser.add_argument('--gamma', default=None, type=float,
                    help='weight of the LiDAR-branch loss (default: dataset-specific)')
parser.add_argument('--lamda', default=None, type=float,
                    help='weight of the fusion-branch loss (default: dataset-specific)')
parser.add_argument('--FM', default=32, type=int,
                    help='number of base feature maps')
parser.add_argument('--runs', default=1, type=int,
                    help='number of repeated runs')
parser.add_argument('--NC', type=int, default=50,
                    help='number of PCA components for HSI dimensionality reduction')

args = parser.parse_args()
config = DATASETS[args.dataset]
# fill in dataset-specific defaults for arguments left unspecified
for key in ('patch_size', 'epoch', 'apha', 'gamma', 'lamda', 'runs'):
    if getattr(args, key) is None:
        setattr(args, key, config[key])

CUDA_DEVICE = torch.device('cuda:0')
hyperparams = vars(args)

# loading data
hsi = sio.loadmat(config['hsi_file'])[config['hsi_key']].astype(np.float32)
lidar = sio.loadmat(config['lidar_file'])[config['lidar_key']].astype(np.float32)
N_CLASSES = config['n_classes']

# preprocessing: per-band min-max normalization, PCA, and symmetric padding
PATCH_SIZE = args.patch_size
padded_size = PATCH_SIZE // 2

[m, n, l] = hsi.shape
for i in range(l):
    minimal = hsi[:, :, i].min()
    maximal = hsi[:, :, i].max()
    hsi[:, :, i] = (hsi[:, :, i] - minimal) / (maximal - minimal)
hsi = np.nan_to_num(hsi)

NC = args.NC
PC = np.reshape(hsi, (m * n, l))
pca = PCA(n_components=NC, copy=True, whiten=False)
PC = pca.fit_transform(PC)
hsi = np.reshape(PC, (m, n, NC))

temp2 = np.pad(hsi[:, :, 0], padded_size, 'symmetric')
[m2, n2] = temp2.shape
padded_hsi = np.empty((m2, n2, NC), dtype='float32')
for i in range(NC):
    padded_hsi[:, :, i] = np.pad(hsi[:, :, i], padded_size, 'symmetric')

if len(lidar.shape) == 2:
    l2 = 1
    lidar = (lidar - lidar.min()) / (lidar.max() - lidar.min())
    lidar = np.nan_to_num(lidar)
    padded_lidar = np.pad(lidar, padded_size, 'symmetric').astype('float32')
else:
    [m, n, l2] = lidar.shape
    padded_lidar = np.empty((m2, n2, l2), dtype='float32')
    for i in range(l2):
        band = lidar[:, :, i]
        band = (band - band.min()) / (band.max() - band.min())
        padded_lidar[:, :, i] = np.pad(np.nan_to_num(band), padded_size, 'symmetric')

hyperparams.update({'n_classes': N_CLASSES, 'cuda_device': CUDA_DEVICE})


def extract_patches(gt):
    """Extract HSI/LiDAR patches and labels for all non-zero pixels of gt."""
    [ind1, ind2] = np.where(gt != 0)
    num = len(ind1)
    patches_hsi = np.empty((num, NC, PATCH_SIZE, PATCH_SIZE), dtype='float32')
    patches_lidar = np.empty((num, l2, PATCH_SIZE, PATCH_SIZE), dtype='float32')
    labels = np.empty(num)
    for i in range(num):
        p = padded_hsi[ind1[i]:ind1[i] + PATCH_SIZE, ind2[i]:ind2[i] + PATCH_SIZE, :]
        patches_hsi[i] = np.transpose(p, (2, 0, 1))
        p = padded_lidar[ind1[i]:ind1[i] + PATCH_SIZE, ind2[i]:ind2[i] + PATCH_SIZE]
        patches_lidar[i] = np.transpose(p.reshape(PATCH_SIZE, PATCH_SIZE, l2), (2, 0, 1))
        labels[i] = gt[ind1[i], ind2[i]] - 1
    return patches_hsi, patches_lidar, labels


def to_onehot(labels, nclasses):
    labels = torch.from_numpy(np.asarray(labels, dtype='int64'))
    onehot = torch.FloatTensor(labels.size(0), nclasses)
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


result_dir = os.path.join('results', args.dataset)
os.makedirs(result_dir, exist_ok=True)

OA, AA_list, Kappa_list = [], [], []
for run in range(args.runs):
    train_map = sio.loadmat(config['train_template'].format(run=run + 1))
    TrainPatch1, TrainPatch2, TrainLabel = extract_patches(train_map['TRLabel'])
    TestPatch1, TestPatch2, TestLabel = extract_patches(train_map['TSLabel'])
    TrainLabel = torch.from_numpy(np.asarray(TrainLabel, dtype='int64'))
    TestLabel = torch.from_numpy(np.asarray(TestLabel, dtype='int64'))
    train_gt_onehot = to_onehot(TrainLabel, N_CLASSES)

    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TestPatch1 = torch.from_numpy(TestPatch1)
    TestPatch2 = torch.from_numpy(TestPatch2)
    TrainPatch = (TrainPatch1, TrainPatch2)
    TestPatch = (TestPatch1, TestPatch2)
    Label = (TrainLabel, TestLabel)

    dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = dataf.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = funcs.SFGF(args.FM, NC, l2, N_CLASSES)
    model.cuda()
    print(model)

    weight_file = os.path.join(result_dir, 'net_params_%s.pkl' % args.dataset)

    start = time.time()
    model, weights, train_loss, test_acc = funcs.train(
        model, train_loader, train_gt_onehot, TrainPatch, TestPatch, Label, hyperparams,
        weight_file=weight_file)
    train_time = time.time() - start

    start = time.time()
    oa, probs, fea, aa, kappa = funcs.val(
        model, weights, TestPatch1, TestPatch2, TestLabel, weight_file=weight_file)
    test_time = time.time() - start

    OA.append(float(oa))
    AA_list.append(aa)
    Kappa_list.append(kappa)
    print('Run %d | OA: %.4f, AA: %.4f, Kappa: %.4f' % (run + 1, oa, aa, kappa))
    print('Training time: %.1fs, Test time: %.1fs' % (train_time, test_time))

results = {'OA': OA, 'AA': AA_list, 'Kappa': Kappa_list}
print('Per-run results:', results)
summary = {
    'mean': {k: float(np.mean(v)) for k, v in results.items()},
    'std': {k: float(np.std(v)) for k, v in results.items()},
}
print('Mean over %d runs | OA: %.4f +- %.4f, AA: %.4f +- %.4f, Kappa: %.4f +- %.4f' % (
    args.runs,
    summary['mean']['OA'], summary['std']['OA'],
    summary['mean']['AA'], summary['std']['AA'],
    summary['mean']['Kappa'], summary['std']['Kappa']))

metrics_file = os.path.join(result_dir, 'metrics.json')
with open(metrics_file, 'w') as f:
    json.dump({'per_run': results, 'mean': summary['mean'], 'std': summary['std']}, f, indent=2)
print('Metrics saved to', metrics_file)
