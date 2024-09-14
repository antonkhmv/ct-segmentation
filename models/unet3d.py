from torch import nn
import torch

import models


def double_conv_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            kernel_size=3,
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=1,
            kernel_size=3,
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class Upsample3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.tail = double_conv_3d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.head(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x2.size()[4]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])

        x1 = torch.cat((x1, x2), dim=1)
        return self.tail(x1)


class Downsample3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = double_conv_3d(in_channels, out_channels)
        self.tail = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.head(x)
        return self.tail(x), x


class Unet3d(models.BaseUnet):
    def __init__(self, hidden_dim, **kwargs):
        in_channels = 1
        out_channels = 1

        down_dims = [in_channels, hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8]
        up_dims = [hidden_dim * 16, hidden_dim * 8, hidden_dim * 4, hidden_dim * 2, hidden_dim]
        bottleneck_dims = down_dims[-1:] + up_dims[:1]

        super().__init__(
            down_layers=nn.ModuleList([Downsample3d(dim1, dim2) for dim1, dim2 in zip(down_dims[:-1], down_dims[1:])]),
            up_layers=nn.ModuleList([Upsample3d(dim1, dim2) for dim1, dim2 in zip(up_dims[:-1], up_dims[1:])]),
            bottleneck_layers=double_conv_3d(bottleneck_dims[0], bottleneck_dims[1]),
            head=nn.Sequential(
                nn.Conv3d(
                    in_channels=up_dims[-1],
                    out_channels=out_channels,
                    kernel_size=1,
                ),
                nn.Sigmoid(),
            )
        )
