import torch.nn as nn

from typing import Union, Callable, Tuple


class FullyConnected(nn.Module):
    def __init__(self, activation: Union[Callable, None], dropout_p: float, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_features)
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        return self.dropout(self.activation(self.batch_norm(self.linear(x))))


class Conv2D(nn.Module):
    def __init__(self, activation: Callable, dropout_p: float, dropout_2d: bool, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int], stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activation
        if dropout_2d:
            self.dropout = nn.Dropout2d(dropout_p)
        else:
            self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.dropout(self.activation(self.batch_norm(self.conv2d(x))))
