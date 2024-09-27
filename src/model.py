from compressai.models import MeanScaleHyperprior
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class MeanScaleHyperpriorGrayscale(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(
            conv(1, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 1)
        )