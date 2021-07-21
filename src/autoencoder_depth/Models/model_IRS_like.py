import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_ as kaiming_normal


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock, self).__init__()

        self.f = nn.Sequential(nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1),
                               nn.BatchNorm2d(n_out), nn.ReLU(inplace=True),
                               nn.Conv2d(n_out, n_out, kernel_size=3, padding=1), nn.BatchNorm2d(n_out))

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.f(x)
        out += residual
        out = torch.relu(out)
        return out


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def predict_flow(in_planes, out_planes=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


class NormNetS(nn.Module):

    def __init__(self, feature_size=1024, in_channel=1, out_channel=1):
        super(NormNetS, self).__init__()

        self.input_channel = in_channel

        # shrink and extract features
        self.conv1 = conv(self.input_channel, 64, 7, 2)
        self.conv2 = ResBlock(64, 128, 2)
        self.conv3 = ResBlock(128, 256, 2)
        self.conv3_1 = ResBlock(256, 256)
        self.conv4 = ResBlock(256, 512, stride=2)
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)
        self.conv6_1 = ResBlock(1024, 1024)

        self.linear_encoder = nn.Linear(1024 * 4 * 6, feature_size)
        self.linear_decoder = nn.Linear(feature_size, 1024 * 4 * 6)
        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.iconv5 = nn.ConvTranspose2d(512, 512, 3, 1, 1)

        self.upconv4 = deconv(512, 256)
        self.iconv4 = nn.ConvTranspose2d(256, 256, 3, 1, 1)

        self.upconv3 = deconv(256, 128)
        self.iconv3 = nn.ConvTranspose2d(128, 128, 3, 1, 1)

        self.upconv2 = deconv(128, 64)
        self.iconv2 = nn.ConvTranspose2d(64, 64, 3, 1, 1)

        self.upconv1 = deconv(64, 32)
        self.iconv1 = nn.ConvTranspose2d(32, 32, 3, 1, 1)

        self.upconv0 = deconv(32, 16)
        self.iconv0 = nn.ConvTranspose2d(16, 16, 3, 1, 1)
        self.pred_flow0 = predict_flow(16, out_channel)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        z = self.linear_encoder(conv6b.view(-1, 1024 * 4 * 6))

        conv6b = self.linear_decoder(z).view(-1, 1024, 4, 6)

        upconv5 = self.upconv5(conv6b)
        iconv5 = self.iconv5(upconv5)

        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(upconv4)

        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(upconv3)

        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(upconv2)

        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(upconv1)

        upconv0 = self.upconv0(iconv1)
        iconv0 = self.iconv0(upconv0)
        pr0 = self.pred_flow0(iconv0)

        if pr0.shape[1] == 3:
            normal = pr0 / (torch.norm(pr0, 2, dim=1, keepdim=True) + 1e-8)
        else:
            normal = torch.tanh(pr0)
        return {'feature': z, "output": normal}

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


if __name__ == '__main__':
    model = NormNetS(1)
    img = torch.randn(16, 1, 240, 376)
    print(model(img).shape)
