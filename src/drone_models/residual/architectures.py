import torch
import torch.nn as nn

from typing import Callable


class RegressionHead(nn.Module):
    def __init__(self, activation: Callable, in_embedding_size: int, unit_size: int, past: int, dropout: float):
        super(RegressionHead, self).__init__()
        self.activation = activation
        self.in_embedding_size = in_embedding_size
        self.unit_size = unit_size
        self.past = past
        self.dropout = nn.Dropout(dropout)

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


class RawEmbedding(nn.Module):
    def __init__(self, activation: Callable, embedding_size: int):
        super(RawEmbedding, self).__init__()
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
        out = self.activation(self.depth_conv2d_0(depth.reshape((-1, 1, 480, 640))))
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
