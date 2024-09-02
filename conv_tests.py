#%%

import torch
import torch.nn as nn

class Conv_simple(nn.Module):
    def __init__(self, upscale=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, (1,1))
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, (3,3), padding=1)
        self.px_shuffle = nn.PixelShuffle(upscale)
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
    
 
    
if __name__ == "__main__":
    model = Conv_simple(upscale=4)
    x = torch.rand([1, 3, 56, 56])
    y = model(x)
    print(x.shape, y.shape)
    
# %%
