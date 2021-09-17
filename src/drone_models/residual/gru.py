import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.learning.run import Run


class GRUResidual(pl.LightningModule, Run):
    def __init__(self, model,
                 past: int,
                 unit_size: int,
                 motor_size: int,
                 activation: Callable,
                 layer: int,
                 embedding: str,
                 optimizer):
        """
        Run with a CNN.
        :param model: The model to use the residual model with for the rollout.
        :param past: The number of past state and motor measures to use.
        :param unit_size: The size of one layer.
        :param motor_size: The number of motors.
        :param activation: The activation function to use.
        :param layer: Number of layer for the neural network.
        :param embedding: Type of embedding to use.
        :param optimizer: lambda x: torch.optim.optimizer(x, params)
        """
        super(GRUResidual, self).__init__()
        self.motor_size = motor_size
        self.n_past = n_past
        self.unit_size = unit_size
        self.num_layers = 2

        self.batch_norm_gru0 = nn.BatchNorm1d(self.n_past)
        self.gru0 = nn.GRU(input_size=13 + self.motor_size,
                           hidden_size=self.unit_size,
                           num_layers=self.num_layers,
                           bias=True)

        self.batch_norm0 = nn.BatchNorm1d(self.unit_size)
        self.linear1 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm1 = nn.BatchNorm1d(self.unit_size)
        self.linear2 = nn.Linear(in_features=self.unit_size, out_features=6, bias=True)
        # self.batch_norm2 = nn.BatchNorm1d(self.unit_size)
        # self.linear3 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        # self.batch_norm3 = nn.BatchNorm1d(self.unit_size)
        # self.linear4 = nn.Linear(in_features=self.unit_size, out_features=6, bias=True)

    def forward(self, motor, state, dt):
        """
        Estimate the residual = GT - model
        :param motor: (batch, step, values=6)
        :param state: (batch, step, values=13)
        :param dt: (batch, step, values=1)
        :return: (batch, value=6)
        """
        state_motor = torch.cat([state, motor, dt], dim=2)

        out = state_motor  # (batch, step=self.n_past, values=20)
        out = self.batch_norm_gru0(out)
        # out: (batch, step, values=20)
        out, _ = self.gru0(out.transpose(0, 1))  # gru takes input with shape: (step, batch, values)
        out = out.transpose(0, 1)  # gru output with shape: (step, batch, values)
        batch_size = out.shape[0]

        out = out.reshape(batch_size * self.n_past, self.unit_size)

        out = self.batch_norm0(out)
        out = relu(self.linear1(out))
        out = self.batch_norm1(out)
        out = self.linear2(out)
        # out = self.batch_norm2(out)
        # out = relu(self.linear3(out))
        # out = self.batch_norm3(out)
        # out = self.linear4(out)
        return out.view(batch_size, self.n_past, 6)  # residual
class CNNResidual(pl.LightningModule, Run):
    def __init__(self, model,
                 past: int,
                 unit_size: int,
                 activation: Callable,
                 layer: int,
                 embedding: str,
                 optimizer):
        """
        Run with a CNN.
        :param model: The model to use the residual model with for the rollout.
        :param past: The number of past state and motor measures to use.
        :param unit_size: The size of one layer.
        :param activation: The activation function to use.
        :param layer: Number of layer for the neural network.
        :param embedding: Type of embedding to use.
        :param optimizer: lambda x: torch.optim.optimizer(x, params)
        """
        super().__init__()
        self.model = model
        self.past = past
        self.unit_size = unit_size
        self.a = activation

        layer_test = [2, 3]
        assert layer in layer_test, f"layer parameter must be one of the following values {layer_test}"
        self.layer = layer

        emb_test = ["none", "depth", "norm", "both"]
        assert embedding in emb_test, f"embedding parameter must be one of the following values {emb_test}"
        self.emb = embedding

        self.optimizer = optimizer

        self.feature_size = 6 + 13
        self.emb_size = 0
        if self.emb in ["depth", "norm"]:
            self.emb_size = 128
        elif self.emb in ["both"]:
            self.emb_size = 2 * 128

        # In shape (batch, channel=1, self.time_size, self.feature_size)
        self.conv2d0 = nn.Conv2d(in_channels=1, out_channels=unit_size, kernel_size=(1, self.feature_size),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")

        self.conv2d1 = nn.Conv2d(in_channels=unit_size, out_channels=unit_size, kernel_size=(1, 1),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")

        self.conv2demb0 = nn.Conv2d(in_channels=1, out_channels=self.unit_size, kernel_size=(1, 1),
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        self.conv2demb1 = nn.Conv2d(in_channels=self.unit_size, out_channels=self.unit_size, kernel_size=(1, 1),
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        self.conv2demb2 = nn.Conv2d(in_channels=1, out_channels=unit_size, kernel_size=(1, self.emb_size),
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # In/Out shape (batch, channel=50, self.time_size, 1)
        self.conv2d_end = nn.Conv2d(in_channels=unit_size, out_channels=6, kernel_size=(self.past, 1),
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # Out shape (batch, channel=6, 1, 1)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def forward(self, motor, state, dt):
        """
        Estimate the residual = GT - model
        :param motor: (batch, step, values=6)
        :param state: (batch, step, values=13, 128+13 or 2*128+13)
        :param dt: (batch, step, values=1)
        :return: (batch, value=6)
        """
        batch_size = state.shape[0]
        state_motor = torch.cat([state, motor], dim=2).view(batch_size, 1, self.past, self.feature_size)
        out = self.a(self.conv2d0(state_motor))
        if self.layer == 3:
            out = self.a(self.conv2d1(out))
        out = self.conv2d_end(out)
        return out.squeeze()  # residual

    def _step(self, batch):
        """
        :param batch: {motor: (batch, time, 6), state: (batch, time, 13), residual: (batch, time, 6)}
        :return: Loss
        """
        motor = batch["motor"]
        state = batch["state"]
        dt = torch.FloatTensor([[[1 / 200]]])
        residual = batch["residual"]
        if self.emb in ["depth", "both"]:
            depth_emb = batch["depth_emb"]
            state = torch.cat([state, depth_emb], dim=-1)
        if self.emb in ["norm", "both"]:
            norm_emb = batch["norm_emb"]
            state = torch.cat([state, norm_emb], dim=-1)

        # Look in past states
        residual_gt = residual[:, -1]

        # Horizon forward
        past_states = state[:, -1 - self.past:-1]
        past_motor = motor[:, -self.past:]
        residual_hat = self.forward(motor=past_motor, state=past_states, dt=dt)

        loss = (residual_gt - residual_hat) ** 2
        return loss.mean()

    def training_step(self, train_batch, batch_idx):
        training_loss = self._step(train_batch)
        self.log("train_loss", training_loss)
        return training_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self._step(val_batch)
        self.log("val_loss", val_loss)

    def rollout(self, motor, state, state_gt, horizon):
        # TODO rework
        dt = torch.FloatTensor([[1 / 200]])
        # state_gt = state[:, -horizon - 1:]

        assert len(state_gt.shape) == 3  # (batch=1, horizon, value=6)
        assert state_gt.shape[1] == horizon + 1

        with torch.no_grad():
            # Prediction rollout
            prediction = [state[:, self.past - 1]]
            for i in range(horizon):
                past_states = torch.cat([state[:, i:self.past - 1],
                                         torch.stack(prediction, dim=1)[:, -self.past:]],
                                        dim=1)
                past_motor = motor[:, i + 1:self.past + i + 1]
                residual = self.forward(state=past_states, motor=past_motor, dt=dt)
                next_state = self.model(motor=past_motor[:, -1], state=past_states[:, -1], dt=dt, residual=residual)
                prediction.append(next_state)

        prediction = torch.stack(prediction, dim=1)
        error = ((prediction - state_gt) ** 2).mean()
        return error, prediction
