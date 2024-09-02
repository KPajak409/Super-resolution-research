# %%
import torch
import torch.nn as nn


class Conv_upsamplex4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 48, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.px_shuffle = nn.PixelShuffle(4)  # C_out = C_in / upscale_factor**2
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.px_shuffle(x)

        return x


class Conv_upsamplex2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 12, (3, 3), padding=1)
        self.px_shuffle = nn.PixelShuffle(2)  # C_out = C_in / upscale_factor**2
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.px_shuffle(x)

        return x


if __name__ == "__main__":
    modelx2 = Conv_upsamplex2()
    modelx4 = Conv_upsamplex4()
    x = torch.rand([1, 3, 56, 56])
    yx2 = modelx2(x)
    yx4 = modelx4(x)
    print(f"Input shape: \t\t{x.shape}")
    print(f"Conv_upsamplex2 out: \t{yx2.shape}")
    print(f"Conv_upsamplex4 out: \t{yx4.shape}")

# %%
