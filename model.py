"""
Frequency-domain cross-modal guided filtering (ALFG & AHFG).

Each generator takes the compressed feature of the guiding modality,
produces a spatially-variant k x k filter kernel at every position, and
applies it to the other modality's feature map (paper Sec. III-C):

- ALFG: adaptive low-pass filter, kernel-wise Softmax (paper Eq. 4)
- AHFG: adaptive high-pass filter, obtained by low-pass inversion
        K_high = E - K_low (paper Eq. 7)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


    def carafe(x, normed_mask, kernel_size, group=1, up=1):
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        assert m_h == up * h
        assert m_w == up * w
        pad = kernel_size // 2
        # reflect padding requires pad < input dim; fall back to replicate for small feature maps
        if pad < min(h, w):
            pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        else:
            pad_x = F.pad(x, pad=[pad] * 4, mode='replicate')
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
        res = unfold_x * normed_mask
        return res.sum(dim=2).reshape(b, c, m_h, m_w)


    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def hamming2D(M, N):
    """Build a 2D Hamming window of shape (M, N) as the outer product of two 1D windows."""
    return np.outer(np.hamming(M), np.hamming(N))


def softmax_kernel_normalizer(mask, kernel, hamming):
    """Normalize raw kernel predictions into per-position k x k filters.

    A kernel-wise Softmax makes all weights non-negative and sum to one
    (the two constraints of a valid low-pass filter), followed by
    Hamming-window regularization.
    """
    n, mask_c, h, w = mask.size()
    mask_channel = int(mask_c / float(kernel ** 2))
    mask = mask.view(n, mask_channel, -1, h, w)
    mask = F.softmax(mask, dim=2, dtype=mask.dtype)
    mask = mask.view(n, mask_channel, kernel, kernel, h, w)
    mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
    mask = mask * hamming
    mask /= mask.sum(dim=(-1, -2), keepdims=True)
    mask = mask.view(n, mask_channel, h, w, -1)
    mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
    return mask


class ALFG(nn.Module):
    """Adaptive Low-pass Filter generator with cross-modal guided filtering.

    `guide_comp` is the compressed feature of the guiding modality; the
    generated low-pass filters are applied to `target_feat` of the other
    modality, suppressing high-frequency noise and enhancing regional
    consistency in a cross-modal manner.
    """

    def __init__(self, compressed_channels=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.generator = nn.Conv2d(compressed_channels, kernel_size ** 2, kernel_size,
                                   padding=kernel_size // 2)
        self.register_buffer('hamming', torch.FloatTensor(
            hamming2D(kernel_size, kernel_size))[None, None])
        xavier_init(self.generator, distribution='uniform')
        normal_init(self.generator, std=0.001)

    def forward(self, guide_comp, target_feat):
        mask = softmax_kernel_normalizer(self.generator(guide_comp), self.kernel_size, self.hamming)
        return carafe(target_feat, mask, self.kernel_size, group=1, up=1)


class AHFG(ALFG):
    """Adaptive High-pass Filter generator with cross-modal guided filtering.

    Follows the low-pass inversion mechanism (paper Eq. 7): the generated
    kernel is normalized into a low-pass filter K_low, and the filtered
    result is (E - K_low) applied to `target_feat`, which removes slowly
    varying components while enhancing edges and structural details.
    """

    def forward(self, guide_comp, target_feat):
        mask = softmax_kernel_normalizer(self.generator(guide_comp), self.kernel_size, self.hamming)
        return target_feat - carafe(target_feat, mask, self.kernel_size, group=1, up=1)
