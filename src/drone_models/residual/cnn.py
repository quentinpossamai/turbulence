import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Callable

from src.learning.run import Run
from src.log import Logger


class CNNResidual(nn.Module, Run):
    def __init__(self, model,
                 past: int,
                 unit_size: int,
                 activation: Callable,
                 layer: int,
                 embedding: str,
                 device,
                 log: Logger = None):
        """
        Run with a CNN.
        :param model: The model to use the residual model with for the rollout.
        :param past: The number of past state and motor measures to use.
        :param unit_size: The size of one layer.
        :param activation: The activation function to use.
        :param layer: Number of layer for the neural network.
        :param embedding: Type of embedding to use.
        :param device: Cuda device to use.
        :param log: The logger to log training and validation error.
        """
        super().__init__()
        self.model = model
        self.past = past
        self.unit_size = unit_size
        self.a = activation
        assert layer in [2, 3]
        self.layer = layer
        self.device = device
        assert embedding in [None, "depth", "norm", "both"]
        self.emb = embedding
        self.log = log

        self.optimizer = None
        self.motor_size = 6
        self.feature_size = 19
        if self.emb in ["depth", "norm"]:
            self.feature_size = self.feature_size + 128
        elif self.emb in ["both"]:
            self.feature_size = self.feature_size + 2 * 128

        # In shape (batch, channel=1, self.time_size, self.feature_size)
        self.conv2d0 = nn.Conv2d(in_channels=1, out_channels=unit_size, kernel_size=(1, self.feature_size),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        if self.layer == 3:
            self.conv2d1 = nn.Conv2d(in_channels=unit_size, out_channels=unit_size, kernel_size=(1, 1),
                                     stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # In/Out shape (batch, channel=50, self.time_size, 1)
        self.conv2d_end = nn.Conv2d(in_channels=unit_size, out_channels=6, kernel_size=(self.past, 1),
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # Out shape (batch, channel=6, 1, 1)

    def configure_optimizers(self, optimizer):
        return optimizer(self.parameters())

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
        motor = batch["motor"].to(self.device)
        state = batch["state"].to(self.device)
        dt = torch.FloatTensor([[[1 / 200]]])
        residual = batch["residual"].to(self.device)
        if self.emb in ["depth", "both"]:
            depth_emb = batch["depth_emb"].to(self.device)
            state = torch.cat([state, depth_emb], dim=-1)
        if self.emb in ["norm", "both"]:
            norm_emb = batch["norm_emb"].to(self.device)
            state = torch.cat([state, norm_emb], dim=-1)

        # Look in past states
        residual_gt = residual[:, -1].to(self.device)

        # Horizon forward
        past_states = state[:, -1 - self.past:-1]
        past_motor = motor[:, -self.past:]
        residual_hat = self.forward(motor=past_motor, state=past_states, dt=dt)

        loss = (residual_gt - residual_hat) ** 2
        return loss.mean(axis=-1)  # Mean on features and not on batch

    def training_step(self, train_batch):
        self.train()
        training_loss = self._step(train_batch)
        self.log.train_epoch_loss_log.append(training_loss.detach())
        return training_loss

    def validation_step(self, val_batch):
        self.eval()
        with torch.no_grad():
            val_loss = self._step(val_batch)
            self.log.valid_epoch_loss_log.append(val_loss)
        return val_loss

    def rollout(self, motor, state, state_gt, horizon):
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
