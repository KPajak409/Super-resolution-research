# %%
import torch
import torch.nn as nn
import math


class ESPCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
    ):
        out_channels = int(in_channels * (upscale_factor**2))

        super().__init__()
        # feature mapping
        self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), padding=1)

        # sub-pixel convolution
        self.conv3 = nn.Conv2d(32, out_channels, (3, 3), padding=1)
        self.px_shuffle = nn.PixelShuffle(
            upscale_factor
        )  # C_out = C_in / upscale_factor**2

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(
                        module.weight.data,
                        0.0,
                        math.sqrt(
                            2 / (module.out_channels * module.weight.data[0][0].numel())
                        ),
                    )
                    nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.px_shuffle(x)

        return x


def espcn_x2(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=2, **kwargs)
    return model


def espcn_x3(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=3, **kwargs)
    return model


def espcn_x4(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=4, **kwargs)
    return model


if __name__ == "__main__":
    modelx2 = espcn_x2(in_channels=3, out_channels=3)
    modelx3 = espcn_x3(in_channels=3, out_channels=3)
    modelx4 = espcn_x4(in_channels=3, out_channels=3)
    x = torch.rand([1, 3, 56, 56])
    yx2 = modelx2(x)
    yx3 = modelx3(x)
    yx4 = modelx4(x)
    print(f"Input shape: \t\t{x.shape}")
    print(f"ESPCNx2 out: \t{yx2.shape}")
    print(f"ESPCNx2 out: \t{yx3.shape}")
    print(f"ESPCNx4 out: \t{yx4.shape}")
