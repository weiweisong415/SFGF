import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

from model import ALFG, AHFG, xavier_init
from spatial import ASFG


class SFGFBlock(nn.Module):
    """One SFGF fusion block (paper Sec. III-C).

    Compresses both modality features and performs cross-modal guided
    filtering in the spatial and frequency domains: filters generated from
    one modality (ASFG / ALFG / AHFG) are applied to the other, and the
    three guided results are combined with the original feature
    (paper Eq. 2).
    """

    def __init__(self, channels, compressed_channels=64, kernel_size=3):
        super().__init__()
        self.hsi_compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.lidar_compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.asfg = ASFG(compressed_channels, kernel_size)
        self.alfg = ALFG(compressed_channels, kernel_size)
        self.ahfg = AHFG(compressed_channels, kernel_size)
        for conv in (self.hsi_compressor, self.lidar_compressor):
            xavier_init(conv, distribution='uniform')

    def forward(self, hsi_feat, lidar_feat):
        hsi_comp = self.hsi_compressor(hsi_feat)
        lidar_comp = self.lidar_compressor(lidar_feat)

        # LiDAR features guided by HSI-generated filters, and vice versa
        lidar_fused = lidar_feat + self.asfg(hsi_comp, lidar_feat) \
                              + self.alfg(hsi_comp, lidar_feat) \
                              + self.ahfg(hsi_comp, lidar_feat)
        hsi_fused = hsi_feat + self.asfg(lidar_comp, hsi_feat) \
                             + self.alfg(lidar_comp, hsi_feat) \
                             + self.ahfg(lidar_comp, hsi_feat)
        return hsi_fused, lidar_fused


class SFGF(nn.Module):
    """Dual-branch CNN with spatial-frequency guidance fusion.

    The HSI and LiDAR branches exchange information through two SFGF blocks
    at multiple scales; the three classifiers (HSI / LiDAR / fused) are
    trained jointly.
    """

    def __init__(self, FM, HNC, LNC, Classes):
        super(SFGF, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=HNC, out_channels=FM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=LNC, out_channels=FM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

        self.sfgf_block2 = SFGFBlock(2 * FM)
        self.sfgf_block3 = SFGFBlock(4 * FM)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y):
        x1 = self.conv1_1(x)
        y1 = self.conv1_2(y)

        x2 = self.conv2(x1)
        y2 = self.conv2(y1)

        x2, y2 = self.sfgf_block2(x2, y2)

        x3 = self.conv3(x2)
        y3 = self.conv3(y2)

        x_fused, y_fused = self.sfgf_block3(x3, y3)

        x4 = self.gap(x_fused).view(x_fused.size(0), -1)
        y4 = self.gap(y_fused).view(y_fused.size(0), -1)

        out1 = self.out1(x4)
        out2 = self.out2(y4)
        out3 = self.out3(x4 + y4)
        return [out1, out2, out3, x4 + y4]


def EncodingOnehot(labels, nclasses):
    labels = torch.from_numpy(np.asarray(labels, dtype='int64'))
    target_onehot = torch.FloatTensor(labels.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, labels.view(-1, 1), 1)
    return target_onehot


def evaluate(net, TestPatch1, TestPatch2, TestLabel, device, test_bs=500):
    """Return predictions of the fused classifier over the test set."""
    net.eval()
    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel) // test_bs
    for i in range(number):
        sem = net(TestPatch1[i * test_bs:(i + 1) * test_bs].to(device),
                  TestPatch2[i * test_bs:(i + 1) * test_bs].to(device))
        output = sem[0] + sem[1] + sem[2]
        pred_y[i * test_bs:(i + 1) * test_bs] = torch.max(output, 1)[1].squeeze().cpu()
    if (i + 1) * test_bs < len(TestLabel):
        sem = net(TestPatch1[(i + 1) * test_bs:len(TestLabel)].to(device),
                  TestPatch2[(i + 1) * test_bs:len(TestLabel)].to(device))
        output = sem[0] + sem[1] + sem[2]
        pred_y[(i + 1) * test_bs:len(TestLabel)] = torch.max(output, 1)[1].squeeze().cpu()
    net.train()
    return torch.from_numpy(pred_y).long()


def train(net, data_loader, train_gt_onehot, TrainPatch, TestPatch, Label, kwargs,
          weight_file='net_params_FusionWeight.pkl'):
    device = kwargs['cuda_device']
    nclasses = kwargs['n_classes']
    epoches = kwargs['epoch']
    apha, gamma, lamda = kwargs['apha'], kwargs['gamma'], kwargs['lamda']
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=kwargs['lr'], weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()

    TrainLabel, TestLabel = Label[0], Label[1]
    TestPatch1, TestPatch2 = TestPatch[0], TestPatch[1]

    test_acc = []
    train_loss = []
    BestAcc = 0
    for e in range(epoches):
        net.train()
        epoch_loss = 0.
        for data_hsi, data_lidar, target_hsi in data_loader:
            data_hsi, data_lidar, target_hsi = data_hsi.cuda(), data_lidar.cuda(), target_hsi.cuda()

            optimizer.zero_grad()
            [semantic_hsi, semantic_lidar, semantic_fusion, _] = net(data_hsi, data_lidar)

            loss = apha * criterion(semantic_hsi, target_hsi) \
                 + gamma * criterion(semantic_lidar, target_hsi) \
                 + lamda * criterion(semantic_fusion, target_hsi)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss.append(epoch_loss / len(data_loader))
        print('Epoch: %d | train loss: %.4f' % (e, train_loss[-1]))

        if (e % kwargs['test_freq'] == 0) | (e == epoches - 1):
            pred_y = evaluate(net, TestPatch1, TestPatch2, TestLabel, device)
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            test_acc.append(accuracy.item())
            print('Epoch: %d | test accuracy: %.4f' % (e, accuracy))
            if accuracy > BestAcc:
                torch.save(net.state_dict(), weight_file)
                BestAcc = accuracy

    weights = [1, 1, 1]
    return net, weights, train_loss, test_acc


def val(net, weights, TestPatch1, TestPatch2, TestLabel, weight_file='net_params_FusionWeight.pkl'):
    """Load the best weights, evaluate on the test set and compute OA / AA / Kappa."""
    net.load_state_dict(torch.load(weight_file))
    net.eval()
    device = 0
    test_bs = 64
    pred_y = np.empty((len(TestLabel)), dtype='float32')
    fea = np.empty((len(TestLabel), 128), dtype='float32')
    number = len(TestLabel) // test_bs
    for i in range(number):
        sem = net(TestPatch1[i * test_bs:(i + 1) * test_bs].cuda(),
                  TestPatch2[i * test_bs:(i + 1) * test_bs].cuda())
        output = sem[0] + sem[1] + sem[2]
        pred_y[i * test_bs:(i + 1) * test_bs] = torch.max(output, 1)[1].squeeze().cpu()
        # fused features (sem[3]) kept for t-SNE visualization
        fea[i * test_bs:(i + 1) * test_bs, :] = sem[3].detach().cpu()
    if (i + 1) * test_bs < len(TestLabel):
        sem = net(TestPatch1[(i + 1) * test_bs:len(TestLabel)].cuda(),
                  TestPatch2[(i + 1) * test_bs:len(TestLabel)].cuda())
        output = sem[0] + sem[1] + sem[2]
        pred_y[(i + 1) * test_bs:len(TestLabel)] = torch.max(output, 1)[1].squeeze().cpu()
        fea[(i + 1) * test_bs:len(TestLabel), :] = sem[3].detach().cpu()

    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

    pred_np = pred_y.numpy()
    true_np = TestLabel.numpy()
    n_classes = int(max(true_np.max(), pred_np.max())) + 1
    aa_list = []
    for c in range(n_classes):
        mask = (true_np == c)
        if mask.sum() > 0:
            aa_list.append(float((pred_np[mask] == c).sum()) / mask.sum())
    AA = float(np.mean(aa_list))
    Kappa = float(cohen_kappa_score(true_np, pred_np))

    return OA, pred_y, fea, AA, Kappa
