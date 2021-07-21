import torch
import torch.nn as nn


# Depth Image is 240x376

class Autoencoder(nn.Module):
    def __init__(self, feature_size=1024, in_channel=1, out_channel=1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(feature_size, in_channel)
        self.decoder = Decoder(feature_size, out_channel)

    def forward(self, x):
        f = self.encoder(x)
        o = self.decoder(f)
        return {'feature': f, "output": o}


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class Encoder(nn.Module):
    def __init__(self, feature_size, channel):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            Block(channel, 16, kernel_size=(7, 7), stride=1, padding=3),
            Block(16, 32, kernel_size=(3, 3), stride=1),
            Block(32, 64, kernel_size=(3, 3), stride=2),
            Block(64, 64, kernel_size=(3, 3), stride=1),
            Block(64, 32, kernel_size=(3, 3), stride=2),
            Block(32, 4, kernel_size=(3, 3), stride=1),
        )

        self.linear = nn.Linear(4 * 60 * 94, feature_size)

    def forward(self, x):
        x = self.net(x)  # [B, 128, 28, 28]
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, feature_size, channel):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(feature_size, 4 * 60 * 94)
        self.net = nn.Sequential(
            Block(4, 128, kernel_size=(3, 3), stride=1),  # 6
            Block(128, 128, kernel_size=(3, 3), stride=1),
            Block(128, 64, kernel_size=(3, 3), stride=1),  # 5
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 32, kernel_size=(7, 7), stride=1, padding=3),  # 1
            nn.Conv2d(32, channel, 1)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 4, 60, 94)
        x = self.net(x)
        if x.shape[1] == 3:
            return x / (torch.norm(x, 2, dim=1, keepdim=True) + 1e-8)
        else:
            return torch.tanh(x)


if __name__ == '__main__':
    depthmap = torch.randn(16, 1, 240, 376)
model = Autoencoder()

y = model(depthmap)
print(y["feature"].shape, y["output"].shape)
