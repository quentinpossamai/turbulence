import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Union, Callable, Tuple

from src.drone_models.model_structure import Run
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel


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


class VariantA(pl.LightningModule, Run):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariantA")
        parser.add_argument('--activation', default="tanh", type=str, help="The activation function to use")
        parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
        parser.add_argument('--past', default=40, type=int, help="Number of past information to take in the model")
        parser.add_argument('--unit_size', default=50, type=int, help="Number of unit for the hidden layers")
        return parent_parser

    def __init__(self,
                 activation: str,
                 learning_rate: float,
                 model,
                 motor_size: int,
                 past: int,
                 unit_size: int,
                 ):
        """
        Model to predict residual f_a = f_aero. Takes motor, state and image from past timesteps.

        param activation: The activation function to use.

        param learning_rate: Learning rate for Adam optimizer.

        param model: The model to use the residual model with for the rollout.

        param motor_size: The number of motors.

        param past: The number of past state and motor measures to use.

        param unit_size: The size of one layer.

        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu

        self.learning_rate = learning_rate
        self.model = model
        self.motor_size = motor_size
        self.past = past
        self.unit_size = unit_size

        self.depth_input_size = [1, 480, 640]
        self.feature_size = self.motor_size + 13

        # self.save_hyperparameters()  # TODO ?

        # - sm (state, motor) branch
        # For the simulation past motor and state is useless
        self.sm_input_norm = nn.BatchNorm1d(self.feature_size)
        self.sm_input_drop = nn.Dropout(p=0.2)
        self.sm_fc_out = FullyConnected(self.activation, dropout_p=0.5, in_features=self.feature_size, out_features=128)

        # - depth (encoding depth) branch
        self.depth_input_norm = nn.BatchNorm2d(1)  # Nbatch, Channel, Height, Width
        # self.depth_input_drop = nn.Dropout2d(0.2)  # no dropout2d for input because dropout2d clears entire channel
        # and input channel number == 1

        self.depth_conv2d_0 = Conv2D(self.activation, 0.5, True, 1, 16, (7, 7), 1, 3)
        self.depth_pool_0 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_1 = Conv2D(self.activation, 0.5, True, 16, 32, (3, 3), 1)
        self.depth_pool_1 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_2 = Conv2D(self.activation, 0.5, True, 32, 64, (3, 3), 2)
        self.depth_pool_2 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_3 = Conv2D(self.activation, 0.5, True, 64, 64, (3, 3), 1)
        self.depth_pool_3 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_4 = Conv2D(self.activation, 0.5, True, 64, 32, (3, 3), 2)
        self.depth_pool_4 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_5 = Conv2D(self.activation, 0.5, True, 32, 4, (3, 3), 1)
        self.depth_pool_5 = nn.MaxPool2d((2, 2))
        self.depth_emb_linear = FullyConnected(None, dropout_p=0.5, in_features=4 * 13 * 18, out_features=128)
        # Time reduction
        self.depth_gru = nn.GRU(input_size=128, hidden_size=128, num_layers=3, batch_first=True)
        self.depth_gru_norm = nn.BatchNorm1d(128)
        self.depth_gru_dropout = nn.Dropout(p=0.5)

        # - reg (regression) branch
        self.reg_fc_0 = FullyConnected(self.activation, 0.5, 128, 50)
        self.reg_fc_1 = FullyConnected(self.activation, 0.5, 50, 50)
        self.reg_linear_out = nn.Linear(in_features=50, out_features=1)

        self.example_input_array = [torch.zeros(2, self.motor_size),
                                    torch.zeros(2, 13),
                                    torch.zeros(2, self.past, 480, 640)]

    def configure_optimizers(self):
        print(f"Lightning.device: {self.device}", flush=True)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, motor: torch.Tensor, state: torch.Tensor, depth: torch.Tensor):
        r"""
        Estimate the residual

        param motor: (*, 6)

        param state: (*, 13)

        param depth: (*, self.past, 480, 640)

        Generally * means batch dimension or more but must exist during training to compute batch_norm.

        return: (*, value=1)
        """
        # - Input processing
        assert motor.shape[-1] == self.motor_size
        assert state.shape[-1] == 13
        assert depth.shape[-3:] == (self.past, 480, 640)
        assert motor.shape[:-1] == state.shape[:-1] == depth.shape[:-3]
        depth_shape = depth.shape

        # - sm branch
        out = torch.cat([motor, state], dim=-1)
        out = self.sm_input_norm(out.reshape(-1, self.feature_size))
        out = self.sm_input_drop(out)
        sm_out = self.sm_fc_out(out)

        # - depth branch
        star_shape = list(depth.shape[:-3])
        out = self.depth_input_norm(depth.reshape((-1, 1, 480, 640)))
        out = self.depth_conv2d_0(out)
        out = self.depth_pool_0(out)
        out = self.depth_conv2d_1(out)
        out = self.depth_pool_1(out)
        out = self.depth_conv2d_2(out)
        # out = self.depth_pool_2(out)
        out = self.depth_conv2d_3(out)
        # out = self.depth_pool_3(out)
        out = self.depth_conv2d_4(out)
        # out = self.depth_pool_4(out)
        out = self.depth_conv2d_5(out)
        out = self.depth_pool_5(out)
        out = self.depth_emb_linear(out.reshape((out.shape[0], -1)))
        out, _ = self.depth_gru(out.reshape((-1, self.past, 128)))  # gru output with shape: (*, step, feature_size)
        out = out[..., -1, :]  # last hidden states, (*, feature_size)
        depth_out = self.depth_gru_dropout(self.depth_gru_norm(out))  # (batch, feature_size)

        # - reg branch
        out = self.reg_fc_0(sm_out + depth_out)
        out = self.reg_fc_1(out)
        out = self.reg_linear_out(out)
        return out

    def step(self, batch):
        """
        param batch: {motor: (batch, 6),
                      state: (batch, 13),
                      depth: (batch, self.past, 480, 640),
                      residual: (batch, 1)}

        return: Loss
        """
        residual_hat = self.forward(motor=batch['motor'], state=batch['state'], depth=batch['depth'])
        residual_gt = batch["ground_force"]

        loss = (residual_gt - residual_hat) ** 2
        return loss.mean()

    def training_step(self, train_batch, batch_idx):
        training_loss = self.step(train_batch)
        self.log("train_loss", training_loss)
        return training_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self.step(val_batch)
        self.log("val_loss", val_loss)

    def rollout(self, **kwargs):  # TODO
        """
        kwargs: motor: (*, past + horizon, motor_size). t+1

        kwargs: state: (*, past + horizon, 13). t

        kwargs: depth_emb: (batch, past+horizon, values=128) t

        kwargs: norm_emb: (batch, past+horizon, values=128) t

        kwargs: embedding: (*, past + horizon, 128 or 256)

        kwargs: dt: (*, past + horizon, 1). t+1 - t

        kwargs: horizon: int

        :return: (*, horizon, 13). state t+1
        """
        # Input processing
        motor = kwargs['motor']
        state = kwargs['state']
        dt = kwargs['dt']
        depth_emb = kwargs['depth_emb']
        norm_emb = kwargs['norm_emb']
        horizon = kwargs['horizon']

        time_size = horizon + self.past
        assert motor.shape[-1] == self.motor_size, f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.reshape(-1, time_size, self.motor_size)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.reshape(-1, time_size, 13)

        assert depth_emb.shape[-1] == 128, "depth_emb len is not 128"
        depth_emb_shape = depth_emb.shape
        depth_emb = depth_emb.reshape(-1, time_size, 128)

        assert norm_emb.shape[-1] == 128, "norm_emb len is not 128"
        norm_emb_shape = norm_emb.shape
        norm_emb = norm_emb.reshape(-1, time_size, 128)

        assert dt.shape[-1] == 1, "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.reshape(-1, time_size, 1)

        pred_shape = list(state_shape)
        pred_shape[-2] = horizon

        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == depth_emb_shape[:-1]
        assert motor_shape[:-1] == norm_emb_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]
        assert time_size == motor_shape[-2]
        # Input processing end

        next_state_gt = state[:, self.past:]

        with torch.no_grad():
            self.eval()
            # Prediction rollout
            prediction = [state[:, self.past - 1].view(-1, 1, 13)]
            past_state = state[:, 0:self.past]
            for i in range(horizon):
                residual = self.forward(motor=motor[:, i + 1:self.past + i + 1],
                                        state=past_state,
                                        depth_emb=depth_emb[:, i:self.past + i],
                                        norm_emb=norm_emb[:, i:self.past + i],
                                        dt=dt[:, i + 1:self.past + i + 1])
                next_state = self.model(motor=motor[:, i + 1],
                                        state=prediction[-1].view(-1, 13),
                                        dt=dt[:, i + 1],
                                        residual=residual).view(-1, 1, 13)
                prediction.append(next_state)
                past_state = torch.cat([past_state[:, 1:], next_state], dim=1)

        prediction = torch.cat(prediction, dim=1)[:, 1:]  # First is not a prediction
        error = ((prediction - next_state_gt) ** 2)
        return error.mean(), error.reshape(pred_shape), prediction.reshape(pred_shape)


def _test():
    dtype = torch.float32
    batch_size = 1000
    past = 60
    horizon = 30
    time_size = past + horizon
    motor = torch.randn(batch_size, time_size, 6, dtype=dtype)
    state = torch.randn(batch_size, time_size, 13, dtype=dtype)
    dt = torch.tensor(1 / 200, dtype=dtype).repeat(batch_size, time_size, 1)
    residual = torch.randn(batch_size, time_size, 6, dtype=dtype)
    depth_emb = torch.randn(batch_size, time_size, 128, dtype=dtype)
    norm_emb = torch.randn(batch_size, time_size, 128, dtype=dtype)

    batch = {"motor": motor, "state": state, "residual": residual, "depth_emb": depth_emb, "norm_emb": norm_emb}

    model = VariantA(
        activation='tanh',
        embedding="both",
        layer=3,
        learning_rate=1e-3,
        model=DynamicalModel(param="identification", dtype=torch.float32),
        motor_size=6,
        past=60,
        unit_size=50, )
    model.configure_optimizers()

    res = model.forward(motor=motor[:, :past], state=state[:, :past], depth_emb=depth_emb[:, :past],
                        norm_emb=norm_emb[:, :past], dt=dt[:, :past])
    assert res.shape == (batch_size, 6)

    res = model.step(batch=batch)
    assert type(res.item()) == float

    # res = model.training_step(train_batch=batch, batch_idx=3)
    # assert type(res.item()) == float
    #
    # res = model.validation_step(val_batch=batch, batch_idx=3)
    # assert res is None

    error, error_tab, res = model.rollout(motor=motor, state=state, depth_emb=depth_emb, norm_emb=norm_emb,
                                          dt=dt, horizon=horizon)
    assert res.shape == (batch_size, time_size - past, 13)


def main():
    _test()


if __name__ == '__main__':
    main()
