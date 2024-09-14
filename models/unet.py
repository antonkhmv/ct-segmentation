import torch.nn as nn


class BaseUnet(nn.Module):
    def __init__(
            self,
            down_layers: nn.ModuleList,
            up_layers: nn.ModuleList,
            bottleneck_layers: nn.Module,
            head: nn.Module,
    ):
        super().__init__()
        assert len(up_layers) == len(down_layers), \
            f"Different number of upscale and downscale layers: {len(up_layers)} != {len(down_layers)}"
        self.down_layers = down_layers
        self.up_layers = up_layers
        self.bottleneck_layers = bottleneck_layers
        self.head = head
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                module.weight = nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        skip_connections = []

        for layer in self.down_layers:
            x, skip = layer(x)
            skip_connections.append(skip)

        x = self.bottleneck_layers(x)

        for layer in self.up_layers:
            skip = skip_connections.pop()
            x = layer(x, skip)

        out = self.head(x)

        return out
