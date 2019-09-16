import torch.nn as nn

class fan_squeeze(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 5, 1, 2)
        self.layers = nn.Sequential(
                                    nn.Conv2d(32, 64, 3, 1, 1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(64, 128, 3, 1, 1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(128, 68, 3, 1, 1),
                                    nn.LeakyReLU(),
                                    )

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        return out

