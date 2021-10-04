import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Optional

from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel


class PresenceClassifier(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GRUResidual")
        parser.add_argument('--activation', default="tanh", type=str, help="The activation function to use")
        parser.add_argument('--embedding', default="both", type=str, help="Type of embedding to use")
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

        assert embedding in ['depth', 'norm', 'both']
        self.embedding = embedding
        if self.embedding in ["depth", "norm"]:
            self.emb_size = 128
        else:
            assert self.embedding in ["both"]
            self.emb_size = 2 * 128

        layer_test = [2, 3]
        assert layer in layer_test, f"layer parameter must be one of the following values {layer_test}"
        self.layer = layer

        self.learning_rate = learning_rate
        self.motor_size = motor_size
        self.past = past
        self.unit_size = unit_size

        self.save_hyperparameters(
            "activation",
            "embedding",
            "layer",
            "learning_rate",
            "motor_size",
            "past",
            "unit_size"
        )

        self.batch_norm0 = nn.BatchNorm1d(self.past * self.emb_size)
        self.gru0 = nn.GRU(input_size=self.emb_size,
                           hidden_size=self.unit_size,
                           num_layers=self.layer,
                           bias=True,
                           batch_first=True)

        self.batch_norm1 = nn.BatchNorm1d(self.unit_size)
        self.linear1 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm2 = nn.BatchNorm1d(self.unit_size)
        self.linear2 = nn.Linear(in_features=self.unit_size, out_features=1, bias=True)

        self.example_input_array = [torch.zeros(10, self.past, 1),
                                    torch.zeros(10, self.past, 128),
                                    torch.zeros(10, self.past, 128)]

    def configure_optimizers(self):
        print(f"Lightning.device: {self.device}")
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, dt: torch.Tensor,
                depth_emb: Optional[torch.Tensor] = None,
                norm_emb: Optional[torch.Tensor] = None):
        """
        Estimate the residual = GT - model

        :param dt: (batch, self.past, values=1)

        :param depth_emb: (batch, self.past, values=128)

        :param norm_emb: (batch, self.past, values=128)

        :return: (batch, value=6)
        """
        # 1.-Input processing
        if depth_emb is not None:
            assert depth_emb.shape[-1] == 128, "depth_emb len is not 128"
            depth_emb_shape = depth_emb.shape
            depth_emb = depth_emb.reshape(-1, self.past, 128)
            embedding = depth_emb
        if norm_emb is not None:
            assert norm_emb.shape[-1] == 128, "norm_emb len is not 128"
            norm_emb_shape = norm_emb.shape
            norm_emb = norm_emb.reshape(-1, self.past, 128)
            embedding = norm_emb
        if self.embedding in ['both']:
            assert depth_emb_shape[:-1] == norm_emb_shape[:-1]
            embedding = torch.cat([depth_emb, norm_emb], dim=-1)
        assert embedding.shape[-1] == self.emb_size, f"embedding len is not {self.emb_size}"

        # 2.1-GRU
        out = self.batch_norm0(embedding.reshape(-1, self.past * self.emb_size))
        out, _ = self.gru0(out.reshape(-1, self.past, self.emb_size))
        # self.unit_size)
        out = out[:, -1]  # last hidden states, (batch, self.unit_size)
        # Fully connected
        out = self.batch_norm1(out)
        out = self.activation(self.linear1(out))
        out = self.batch_norm2(out)
        out = torch.softmax(self.linear2(out), dim=1)

        out_shape = list(embedding.shape)
        out_shape[-1] = 1
        del out_shape[-2]
        return out.reshape(out_shape)

    def step(self, batch):
        """
        :param batch: {
                       depth_emb: (batch, time, 128),
                       norm_emb: (batch, time, 128),
                       fa_label: (batch, time, 1)}

        :return: Loss
        """
        depth_emb = batch["depth_emb"][:, :self.past]
        norm_emb = batch["norm_emb"][:, :self.past]
        fa_label_gt = batch["fa_label"][:, self.past - 1]

        depth_emb_shape = depth_emb.shape
        dt_shape = list(depth_emb_shape)
        dt_shape[-1] = 1
        dt = torch.tensor(1 / 200).type_as(depth_emb).repeat(dt_shape)

        fa_label_hat = self.forward(depth_emb=depth_emb, norm_emb=norm_emb, dt=dt)

        weights = fa_label_gt * 0.98 + 0.01
        # loss_object = torch.nn.BCELoss(weights)
        # loss = loss_object(fa_label_hat, fa_label_gt)
        return 1

    def training_step(self, train_batch, batch_idx):
        training_loss = self.step(train_batch)
        self.log("train_loss", training_loss)
        return training_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self.step(val_batch)
        self.log("val_loss", val_loss)


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
