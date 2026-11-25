"""
Spatial-domain cross-modal guided filtering (ASFG).

The ASFG (Adaptive Spatial-semantic Filter generator, paper Sec. III-C1)
predicts spatially-variant k x k filters from the compressed feature of one
modality and applies them to the other modality's feature map, capturing
cross-modal spatial-semantic correlations and enabling the interaction of
structural information between the two heterogeneous modalities.
"""

import torch
import torch.nn as nn

from model import carafe, hamming2D, xavier_init


def sigmoid_kernel_normalizer(mask, kernel, hamming):
    """Normalize raw kernel predictions into per-position k x k filters.

    Unlike the low-pass case, no Softmax is applied (the spatial-semantic
    filter is not constrained to be an averaging operator): a Sigmoid keeps
    the weights non-negative and smooth, followed by L1 normalization over
    each kernel and Hamming-window regularization.
    """
    n, mask_c, h, w = mask.size()
    mask_channel = int(mask_c / float(kernel ** 2))
    mask = mask.view(n, mask_channel, -1, h, w)
    mask = torch.sigmoid(mask)
    mask = mask / (mask.sum(dim=2, keepdim=True) + 1e-8)
    mask = mask.view(n, mask_channel, kernel, kernel, h, w)
    mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
    mask = mask * hamming
    mask /= mask.sum(dim=(-1, -2), keepdims=True)
    mask = mask.view(n, mask_channel, h, w, -1)
    mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
    return mask


class ASFG(nn.Module):
    """Adaptive Spatial-semantic Filter generator with cross-modal guided filtering.

    `guide_comp` is the compressed feature of the guiding modality; the
    generated spatial-semantic filters are applied to `target_feat` of the
    other modality, adaptively fusing complementary spatial structure.
    """

    def __init__(self, compressed_channels=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.generator = nn.Conv2d(compressed_channels, kernel_size ** 2, kernel_size,
                                   padding=kernel_size // 2)
        self.register_buffer('hamming', torch.FloatTensor(
            hamming2D(kernel_size, kernel_size))[None, None])
        xavier_init(self.generator, distribution='uniform')

    def forward(self, guide_comp, target_feat):
        mask = sigmoid_kernel_normalizer(self.generator(guide_comp), self.kernel_size, self.hamming)
        return carafe(target_feat, mask, self.kernel_size, group=1, up=1)
