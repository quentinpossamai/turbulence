import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Optional

from src.drone_models.model_structure import Run
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel


class GRUCNNResidual(pl.LightningModule, Run):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GRUResidual")
        parser.add_argument('--activation', default="tanh", type=str, help="The activation function to use")

        def none_or_str(value):
            return None if value == 'None' else value

        parser.add_argument('--embedding', default="depth", type=none_or_str, help="Type of embedding to use")
        parser.add_argument('--horizon', default=1, type=int, help="Number of time step to predict by rollout")
        parser.add_argument('--layer', default=3, type=int, help="Number of layer in the neural network")
        parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
        parser.add_argument('--past', default=60, type=int, help="Number of past information to take in the model")
        parser.add_argument('--unit_size', default=50, type=int, help="Number of unit for the hidden layers")
        return parent_parser

    def __init__(self,
                 activation: str,
                 embedding: str,
                 layer: int,
                 learning_rate: float,
                 model,
                 motor_size: int,
                 past: int,
                 unit_size: int,
                 ):
        """
        Run with a CNN.

        :param activation: The activation function to use.

        :param embedding: Type of embedding to use.

        :param layer: Number of layer for the neural network.

        :param learning_rate: Learning rate for Adam optimizer.

        :param model: The model to use the residual model with for the rollout.

        :param motor_size: The number of motors.

        :param past: The number of past state and motor measures to use.

        :param unit_size: The size of one layer.

        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu

        emb_test = [None, "depth", "norm", "both"]
        assert embedding in emb_test, f"embedding parameter must be one of the following values {emb_test}"
        self.embedding = embedding
        self.emb_size = 0
        if self.embedding in ["depth", "norm"]:
            self.emb_size = 128
        elif self.embedding in ["both"]:
            self.emb_size = 2 * 128

        layer_test = [2, 3]
        assert layer in layer_test, f"layer parameter must be one of the following values {layer_test}"
        self.layer = layer

        self.learning_rate = learning_rate
        self.model = model
        self.motor_size = motor_size
        self.past = past
        self.unit_size = unit_size
        self.feature_size = self.motor_size + 13 + 1

        # self.save_hyperparameters()

        self.batch_norm_input = nn.BatchNorm1d(self.past * self.feature_size)
        self.dropout_input = nn.Dropout(p=0.2)
        self.gru0 = nn.GRU(input_size=self.feature_size,
                           hidden_size=self.unit_size,
                           num_layers=self.layer,
                           bias=True,
                           batch_first=True)
        self.batch_norm_gru = nn.BatchNorm1d(self.unit_size)
        self.dropout_gru = nn.Dropout(p=0.5)

        if self.emb_size != 0:
            self.emb_batch_norm_input = nn.BatchNorm1d(self.emb_size)
            self.emb_dropout_input = nn.Dropout(p=0.2)

            self.emb_linear0 = nn.Linear(in_features=self.emb_size, out_features=self.unit_size, bias=True)
            self.emb_batch_norm0 = nn.BatchNorm1d(self.unit_size)
            self.emb_dropout0 = nn.Dropout(p=0.5)

            self.emb_linear1 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
            self.emb_batch_norm1 = nn.BatchNorm1d(self.unit_size)
            self.emb_dropout1 = nn.Dropout(p=0.5)

            self.linear0 = nn.Linear(in_features=2 * self.unit_size, out_features=self.unit_size, bias=True)
        else:
            self.linear0 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm0 = nn.BatchNorm1d(self.unit_size)
        self.dropout0 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(in_features=self.unit_size, out_features=6, bias=True)

        self.example_input_array = [torch.zeros(10, self.past, self.motor_size),
                                    torch.zeros(10, self.past, 13),
                                    torch.zeros(10, self.past, 1),
                                    torch.zeros(10, 128),
                                    torch.zeros(10, 128)]

    def configure_optimizers(self):
        print(f"Lightning.device: {self.device}")
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, motor: torch.Tensor,
                state: torch.Tensor,
                dt: torch.Tensor,
                depth_emb: Optional[torch.Tensor] = None,
                norm_emb: Optional[torch.Tensor] = None):
        """
        Estimate the residual = GT - model

        :param motor: (batch, self.past, values=6)

        :param state: (batch, self.past, values=13)

        :param dt: (batch, self.past, values=1)

        :param depth_emb: (batch, self.past, values=128)

        :param norm_emb: (batch, self.past, values=128)

        :return: (batch, value=6)
        """
        # 1.-Input processing
        assert motor.shape[-1] == self.motor_size, f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.reshape(-1, self.past, self.motor_size)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.reshape(-1, self.past, 13)

        assert dt.shape[-1] == 1, "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.reshape(-1, self.past, 1)

        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]

        state_motor_dt = torch.cat([motor, state, dt], dim=2)

        # 2.1-GRU state-motor-dt
        out = state_motor_dt  # (batch, self.n_past, feature_size)
        out = self.batch_norm_input(out.reshape(-1, self.past * self.feature_size))
        out = self.dropout_input(out.reshape(-1, self.past, self.feature_size))
        out, _ = self.gru0(out)  # gru output with shape: (batch, step, feature_size)
        out = out[:, -1]  # last hidden states, (batch, feature_size)
        out = self.dropout_gru(self.batch_norm_gru(out))  # (batch, feature_size)

        # 2.2-CNN embedding
        if self.embedding is not None:
            if self.embedding in ["depth", "both"]:
                assert depth_emb.shape[-1] == 128, "depth_emb len is not 128"
                depth_emb_shape = depth_emb.shape
                depth_emb = depth_emb.reshape(-1, 128)
                assert motor_shape[:-2] == depth_emb_shape[:-1]
                embedding = depth_emb
            if self.embedding in ["norm", "both"]:
                assert norm_emb.shape[-1] == 128, "norm_emb len is not 128"
                norm_emb_shape = norm_emb.shape
                norm_emb = norm_emb.reshape(-1, 128)
                assert motor_shape[:-2] == norm_emb_shape[:-1]
                embedding = norm_emb
            if self.embedding in ['both']:
                embedding = torch.cat([depth_emb, norm_emb], dim=-1)
            assert embedding.shape[-1] == self.emb_size, f"embedding len is not {self.emb_size}"
            out_emb = self.emb_dropout_input(self.emb_batch_norm_input(embedding))
            out_emb = self.emb_dropout0(self.activation(self.emb_batch_norm0(self.emb_linear0(out_emb))))
            out_emb = self.emb_dropout1(self.activation(self.emb_batch_norm1(self.emb_linear1(out_emb))))

            out = torch.cat([out, out_emb], dim=1)

        out = self.dropout0(self.activation(self.batch_norm0(self.linear0(out))))
        out = self.linear1(out)

        residual_shape = list(state_shape)
        residual_shape[-1] = 6
        del residual_shape[-2]
        return out.reshape(residual_shape)

    def step(self, batch):
        """
        :param batch: {motor: (batch, time, 6),
                       state: (batch, time, 13),
                       residual: (batch, time, 6),
                       depth_emb: (batch, time, 128),
                       norm_emb: (batch, time, 128)}

        :return: Loss
        """
        motor = batch["motor"][:, 1:self.past + 1]
        dt = batch["dt"][:, 1:self.past + 1]
        state = batch["state"][:, :self.past]
        depth_emb = batch["depth_emb"][:, self.past - 1]
        norm_emb = batch["norm_emb"][:, self.past - 1]

        residual_gt = batch["residual"][:, self.past]
        residual_hat = self.forward(motor=motor, state=state, depth_emb=depth_emb, norm_emb=norm_emb, dt=dt)

        loss = (residual_gt - residual_hat) ** 2
        return loss.mean()

    def training_step(self, train_batch, batch_idx):
        training_loss = self.step(train_batch)
        self.log("train_loss", training_loss)
        return training_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self.step(val_batch)
        self.log("val_loss", val_loss)

    # def rollout(self, motor, state, embedding, dt, horizon):
    def rollout(self, **kwargs):
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

    model = GRUCNNResidual(
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
