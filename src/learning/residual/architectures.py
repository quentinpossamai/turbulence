import torch
import torch.nn as nn

from typing import Callable, Tuple


class RegressionHead(nn.Module):
    def __init__(self, activation: Callable, in_embedding_size: int, unit_size: int, past: int, dropout: float):
        super(RegressionHead, self).__init__()
        self.activation = activation
        self.in_embedding_size = in_embedding_size
        self.unit_size = unit_size
        self.past = past
        self.dropout = nn.Dropout(dropout)

        # TODO progressive regression
        self.emb_reduc_linear = nn.Linear(in_features=self.in_embedding_size, out_features=self.unit_size)
        if self.past > 1:
            self.emb_gru = nn.GRU(input_size=self.unit_size, hidden_size=self.unit_size, num_layers=1, batch_first=True)
        self.reg_linear_0 = nn.Linear(self.unit_size, self.unit_size)
        self.reg_linear_1 = nn.Linear(self.unit_size, self.unit_size)
        self.reg_linear_out = nn.Linear(in_features=self.unit_size, out_features=1)

    def forward(self, out):
        out = self.activation(self.emb_reduc_linear(out))
        out = self.dropout(out)

        if self.past > 1:
            out, _ = self.emb_gru(out.reshape((-1, self.past, self.unit_size)))
            # gru output with shape: (*, self.past, self.unit_size)
            out = out[:, -1, :]  # last hidden states, (*, 50)
        emb_out = out  # (batch, 50)

        # - reg branch
        out = self.activation(self.reg_linear_0(emb_out))
        out = self.dropout(out)
        out = self.activation(self.reg_linear_1(out))
        out = self.dropout(out)
        out = self.reg_linear_out(out)
        return out


class CNNEncoder(nn.Module):
    def __init__(self, embedding_size: int, layer_num: int, channel_num: int):
        super(CNNEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.layer_num = layer_num
        self.ch_num = channel_num  # the maximum channel number  64

        if self.layer_num == 3:
            self.encoder_conv = nn.Sequential(
                # Initial is (batch, 1, 480, 640)
                nn.Conv2d(1, self.ch_num // 2, (7, 7), stride=2, padding=3),
                # (batch, ch_num // 2, 240, 320)
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),  # (batch, ch_num // 2, 120, 160)
                nn.Conv2d(self.ch_num // 2, self.ch_num, (3, 3), stride=2),
                # (batch, ch_num, 60, 80)
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),  # (batch, ch_num, 30, 40)
                nn.Conv2d(self.ch_num, 4, (3, 3), stride=2, padding=(0, 1)),
                # (batch, 4, 15, 20)
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),  # (batch, 4, 7, 10)
            )

        elif self.layer_num == 6:
            self.encoder_conv = nn.Sequential(
                # Initial is (batch, 1, 480, 640)
                nn.Conv2d(1, self.ch_num // 4, (7, 7), stride=1, padding=3),  # (batch, ch_num // 4, 480, 640)
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),  # (batch, ch_num // 4, 240, 320)
                nn.Conv2d(self.ch_num // 4, self.ch_num // 2, (3, 3), stride=1, padding=2),
                # (batch, ch_num // 2, 240, 320)
                nn.ReLU(True),
                nn.Conv2d(self.ch_num // 2, self.ch_num, (3, 3), stride=2),  # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.Conv2d(self.ch_num, self.ch_num, (3, 3), stride=1, padding=2),  # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),  # (batch, ch_num, 60, 80)
                nn.Conv2d(self.ch_num, self.ch_num // 2, (3, 3), stride=2),  # (batch, ch_num // 2, 30, 40)
                nn.ReLU(True),
                nn.MaxPool2d((2, 2)),  # (batch, ch_num // 2, 15, 20)
                nn.Conv2d(self.ch_num // 2, 4, (3, 3), stride=2, padding=(0, 1)),  # (batch , 4, 7, 10)
                nn.ReLU(True),
            )

        else:
            ch = torch.linspace(4, self.ch_num, 5).long().numpy()
            assert self.layer_num == 9, "layer_num must be 3 or 6 or 9"
            self.encoder_conv = nn.Sequential(
                # Initial is (batch, 1, 480, 640)
                nn.Conv2d(in_channels=1, out_channels=ch[1], kernel_size=(7, 7), stride=1, padding=3),  # (batch, *, 480, 640)
                nn.ReLU(True),
                nn.Conv2d(ch[1], ch[2], (3, 3), stride=2),  # (batch, *, 240, 320)
                nn.ReLU(True),
                nn.Conv2d(ch[2], ch[3], (3, 3), stride=1, padding=2),  # (batch, *, 240, 320)
                nn.ReLU(True),
                nn.Conv2d(ch[3], ch[4], (3, 3), stride=2),  # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.Conv2d(ch[4], ch[4], (3, 3), stride=1, padding=2),  # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.Conv2d(ch[4], ch[3], (3, 3), stride=2),  # (batch, *, 60, 80)
                nn.ReLU(True),
                nn.Conv2d(ch[3], ch[2], (3, 3), stride=2),  # (batch, *, 30, 40)
                nn.ReLU(True),
                nn.Conv2d(ch[2], ch[1], (3, 3), stride=2),  # (batch, *, 15, 20)
                nn.ReLU(True),
                nn.Conv2d(ch[1], ch[0], (3, 3), stride=2, padding=(1, 1)),  # (batch, 4, 7, 10)
                nn.ReLU(True),
            )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=4 * 7 * 10, out_features=self.embedding_size),  # 280
            nn.ReLU(True),
            nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size),
        )

    def forward(self, x):
        x = self.encoder_conv(x)  # (batch, 4, 7, 10)
        x = self.flatten(x)  # (batch, 280)  `
        x = self.encoder_lin(x)  # (batch, embedding_size)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, embedding_size: int, layer_num: int, channel_num: int, out_dim: Tuple[int, int, int]):
        super(CNNDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.ch_num = channel_num
        self.layer_num = layer_num
        self.out_dim = out_dim

        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size),
            nn.ReLU(True),
            nn.Linear(in_features=self.embedding_size, out_features=4 * 7 * 10),  # 280
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(4, 7, 10))

        if self.layer_num == 3:
            self.decoder_conv = nn.Sequential(
                # Initial is (batch, 4, 7, 10)
                nn.UpsamplingBilinear2d(size=(15, 20)),  # (batch, 4, 15, 20)
                nn.ConvTranspose2d(4, self.ch_num, (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, ch_num, 30, 40)
                nn.ReLU(True),
                nn.UpsamplingBilinear2d(scale_factor=2),  # (batch, ch_num, 60, 80)
                nn.ConvTranspose2d(self.ch_num, self.ch_num // 2, (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, ch_num // 2, 120, 160)
                nn.ReLU(True),
                nn.UpsamplingBilinear2d(scale_factor=2),  # (batch, ch_num // 2, 240, 320)
                nn.ConvTranspose2d(self.ch_num // 2, self.out_dim[2], (7, 7), stride=2, padding=3, output_padding=1),
                # (batch, self.out_dim[2], 480, 640)
            )
        elif self.layer_num == 6:
            self.decoder_conv = nn.Sequential(
                # Initial is (batch, 4, 7, 10)
                nn.ConvTranspose2d(4, self.ch_num // 2, (3, 3), stride=2, padding=(0, 1), output_padding=(0, 1)),
                # (batch, ch_num // 2, 15, 20)
                nn.ReLU(True),
                nn.UpsamplingBilinear2d(scale_factor=2),  # (batch, ch_num // 2, 30, 40)
                nn.ConvTranspose2d(self.ch_num // 2, self.ch_num, (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, ch_num, 60, 80)
                nn.ReLU(True),
                nn.UpsamplingBilinear2d(scale_factor=2),  # (batch, ch_num, 120, 160)
                nn.ConvTranspose2d(self.ch_num, self.ch_num, (3, 3), stride=1, padding=1),  # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.ConvTranspose2d(self.ch_num, self.ch_num // 2, (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, ch_num // 2, 240, 320)
                nn.ReLU(True),
                nn.ConvTranspose2d(self.ch_num // 2, self.ch_num // 4, (3, 3), stride=1, padding=1),
                # (batch, ch_num // 4, 240, 320)
                nn.ReLU(True),
                nn.UpsamplingBilinear2d(scale_factor=2),  # (batch, ch_num // 4, 480, 640)
                nn.ConvTranspose2d(self.ch_num // 4, self.out_dim[2], (7, 7), stride=1, padding=3)
                # (batch, self.out_dim[2], 480, 640)
            )
        else:
            assert self.layer_num == 9
            ch = torch.linspace(4, self.ch_num, 5).long().numpy()
            self.decoder_conv = nn.Sequential(
                # Initial is (batch, 4, 7, 10)
                nn.ConvTranspose2d(ch[0], ch[1], (3, 3), stride=2, padding=(0, 1), output_padding=(0, 1)),
                # (batch, *, 15, 20)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[1], ch[2], (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, *, 30, 40)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[2], ch[3], (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, *, 60, 80)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[3], ch[4], (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[4], ch[4], (3, 3), stride=1, padding=1),  # (batch, ch_num, 120, 160)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[4], ch[3], (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, *, 240, 320)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[3], ch[2], (3, 3), stride=1, padding=1),
                # (batch, *, 240, 320)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[2], ch[1], (3, 3), stride=2, padding=1, output_padding=1),
                # (batch, *, 480, 640)
                nn.ReLU(True),
                nn.ConvTranspose2d(ch[1], self.out_dim[2], (7, 7), stride=1, padding=3)
                # (batch, self.out_dim[2], 480, 640)
            )

    def forward(self, x):
        # embedding with shape (batch, feature_size)
        x = self.decoder_lin(x)  # (batch, 280)
        x = self.unflatten(x)  # (batch, 4, 7, 10)
        x = self.decoder_conv(x)  # (batch, self.out_dim[-1], 480, 640)
        if x.shape[1] == 3:
            x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        return x


class CNNEncoderOld(nn.Module):
    def __init__(self, activation: Callable, embedding_size: int):
        super(CNNEncoderOld, self).__init__()
        self.activation = activation
        self.embedding_size = embedding_size

        # - emb (embedding) branch
        self.depth_conv2d_0 = nn.Conv2d(1, 16, (7, 7), 1, 3)
        self.depth_pool_0 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_1 = nn.Conv2d(16, 32, (3, 3), 1)
        self.depth_conv2d_2 = nn.Conv2d(32, 64, (3, 3), 2)
        self.depth_conv2d_3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.depth_pool_1 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_4 = nn.Conv2d(64, 32, (3, 3), 2)
        self.depth_pool_2 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_5 = nn.Conv2d(32, 4, (3, 3), 1)
        self.depth_pool_3 = nn.MaxPool2d((2, 2))
        self.depth_linear_0 = nn.Linear(in_features=4 * 6 * 8,  # 192
                                        out_features=self.embedding_size)

    def forward(self, depth):
        out = self.activation(self.depth_conv2d_0(depth))
        out = self.depth_pool_0(out)
        out = self.activation(self.depth_conv2d_1(out))
        out = self.activation(self.depth_conv2d_2(out))
        out = self.activation(self.depth_conv2d_3(out))
        out = self.depth_pool_1(out)
        out = self.activation(self.depth_conv2d_4(out))
        out = self.depth_pool_2(out)
        out = self.activation(self.depth_conv2d_5(out))
        out = self.depth_pool_3(out)
        out = self.depth_linear_0(out.reshape((-1, 4 * 6 * 8)))
        return out


class CNNDecoderOld(nn.Module):
    def __init__(self, activation: Callable, embedding_size: int, out_dim: Tuple[int, int, int]):
        super(CNNDecoderOld, self).__init__()
        self.activation = activation
        self.embedding_size = embedding_size

        # - emb (embedding) branch
        self.depth_conv2d_0 = nn.Conv2d(1, 16, (7, 7), 1, 3)
        self.depth_pool_0 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_1 = nn.Conv2d(16, 32, (3, 3), 1)
        self.depth_conv2d_2 = nn.Conv2d(32, 64, (3, 3), 2)
        self.depth_conv2d_3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.depth_pool_1 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_4 = nn.Conv2d(64, 32, (3, 3), 2)
        self.depth_pool_2 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_5 = nn.Conv2d(32, 4, (3, 3), 1)
        self.depth_pool_3 = nn.MaxPool2d((2, 2))
        self.depth_linear_0 = nn.Linear(in_features=4 * 6 * 8,  # 192
                                        out_features=self.embedding_size)

    def forward(self, embedding):
        out = self.activation(self.depth_conv2d_0(embedding))
        out = self.depth_pool_0(out)
        out = self.activation(self.depth_conv2d_1(out))
        out = self.activation(self.depth_conv2d_2(out))
        out = self.activation(self.depth_conv2d_3(out))
        out = self.depth_pool_1(out)
        out = self.activation(self.depth_conv2d_4(out))
        out = self.depth_pool_2(out)
        out = self.activation(self.depth_conv2d_5(out))
        out = self.depth_pool_3(out)
        out = self.depth_linear_0(out.reshape((-1, 4 * 6 * 8)))
        return out
