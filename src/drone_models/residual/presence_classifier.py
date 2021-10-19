import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Optional


class PresenceClassifier(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GRUResidual")
        parser.add_argument('--activation', default="tanh", type=str, help="The activation function to use")
        parser.add_argument('--embedding', default="norm", type=str, help="Type of embedding to use")
        parser.add_argument('--l1_coef', default=0., type=float, help="L1 loss coefficient")
        parser.add_argument('--l2_coef', default=0., type=float, help="L2 loss coefficient")
        parser.add_argument('--layer', default=3, type=int, help="Number of layer in the neural network")
        parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
        parser.add_argument('--unit_size', default=50, type=int, help="Number of unit for the hidden layers")
        return parent_parser

    def __init__(self,
                 activation: str,
                 embedding: str,
                 l1_coef: float,
                 l2_coef: float,
                 layer: int,
                 learning_rate: float,
                 motor_size: int,
                 unit_size: int,
                 ):
        """
        Run with a CNN.

        :param activation: The activation function to use.

        :param embedding: Type of embedding to use.

        :param l1_coef: Coefficient for l1 regularization.

        :param l2_coef: Coefficient for l2 regularization.

        :param layer: Number of layer for the neural network.

        :param learning_rate: Learning rate for Adam optimizer.

        :param motor_size: The number of motors.

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

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        layer_test = [2, 3]
        assert layer in layer_test, f"layer parameter must be one of the following values {layer_test}"
        self.layer = layer

        self.learning_rate = learning_rate
        self.motor_size = motor_size
        self.unit_size = unit_size

        self.save_hyperparameters()

        self.batch_norm_input = nn.BatchNorm1d(self.emb_size)
        self.dropout_input = nn.Dropout(p=0.2)

        self.linear0 = nn.Linear(in_features=self.emb_size, out_features=self.unit_size, bias=True)
        self.batch_norm0 = nn.BatchNorm1d(self.unit_size)
        self.dropout0 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm1 = nn.BatchNorm1d(self.unit_size)
        self.dropout1 = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(in_features=self.unit_size, out_features=1, bias=True)

        self.example_input_array = [torch.zeros(10, 128),
                                    torch.zeros(10, 128)]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, depth_emb: Optional[torch.Tensor] = None, norm_emb: Optional[torch.Tensor] = None):
        """
        Estimate the residual = GT - model

        :param depth_emb: (*, values=128)

        :param norm_emb: (*, values=128)

        :return: (*, values=1)
        """
        # 1.-Input processing
        if self.embedding in ['depth', 'both']:
            assert depth_emb.shape[-1] == 128, "depth_emb len is not 128"
            depth_emb_shape = depth_emb.shape
            depth_emb = depth_emb.reshape(-1, 128)
            embedding = depth_emb
            embedding_shape = embedding.shape
        if self.embedding in ['norm', 'both']:
            assert norm_emb.shape[-1] == 128, "norm_emb len is not 128"
            norm_emb_shape = norm_emb.shape
            norm_emb = norm_emb.reshape(-1, 128)
            embedding = norm_emb
            embedding_shape = embedding.shape
        if self.embedding in ['both']:
            assert depth_emb_shape[:-1] == norm_emb_shape[:-1]
            embedding = torch.cat([depth_emb, norm_emb], dim=-1)
            embedding_shape = embedding.shape
        assert embedding.shape[-1] == self.emb_size, f"embedding len is not {self.emb_size}"

        # 2.1-MLP
        out = self.dropout_input(self.batch_norm_input(embedding))  # Input
        out = self.dropout0(self.activation(self.batch_norm0(self.linear0(out))))  # hidden layer 0
        out = self.dropout1(self.activation(self.batch_norm1(self.linear1(out))))  # hidden layer 1
        out = torch.sigmoid(self.linear2(out))  # output layer

        out_shape = list(embedding_shape)
        out_shape[-1] = 1
        return out.reshape(out_shape)

    def step(self, batch):
        """
        :param batch: {depth_emb: (batch, time, 128),
                       norm_emb: (batch, time, 128),
                       fa_label: (batch, time, 1)}

        :return: Loss
        """
        depth_emb = batch["depth_emb"]
        norm_emb = batch["norm_emb"]
        fa_label_gt = batch["fa_label"]
        fa_label_hat = self.forward(depth_emb=depth_emb, norm_emb=norm_emb)

        # L2 regularization
        l1_loss, l2_loss = 0, 0
        for param in self.parameters():
            l1_loss += param.abs().sum()
            l2_loss += (param ** 2).sum()

        # Weighted BCE loss
        weights = fa_label_gt * 0.975 + 0.025
        loss_object = torch.nn.BCELoss(weights)
        weighted_bce_loss = loss_object(fa_label_hat, fa_label_gt)
        return weighted_bce_loss.mean() + self.l1_coef * l1_loss + self.l2_coef * l2_loss

    def training_step(self, train_batch, batch_idx):
        training_loss = self.step(train_batch)
        self.log("train_loss", training_loss)
        self.log("learning_rate", self.learning_rate)
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
    # dt = torch.tensor(1 / 200, dtype=dtype).repeat(batch_size, time_size, 1)
    residual = torch.randn(batch_size, time_size, 6, dtype=dtype)
    depth_emb = torch.randn(batch_size, time_size, 128, dtype=dtype)
    norm_emb = torch.randn(batch_size, time_size, 128, dtype=dtype)

    batch = {"motor": motor, "state": state, "residual": residual, "depth_emb": depth_emb, "norm_emb": norm_emb}

    model = PresenceClassifier(
        activation='tanh',
        embedding="both",
        l1_coef=0,
        l2_coef=0,
        layer=3,
        learning_rate=1e-3,
        motor_size=6,
        unit_size=50)
    model.configure_optimizers()

    res = model.forward(depth_emb=depth_emb[:, :past], norm_emb=norm_emb[:, :past])
    assert res.shape == (batch_size, 6)

    res = model.step(batch=batch)
    assert type(res.item()) == float

    # res = model.training_step(train_batch=batch, batch_idx=3)
    # assert type(res.item()) == float
    #
    # res = model.validation_step(val_batch=batch, batch_idx=3)
    # assert res is None

    error, error_tab, res = model.rollout(motor=motor, state=state, depth_emb=depth_emb, norm_emb=norm_emb,
                                          horizon=horizon)
    assert res.shape == (batch_size, time_size - past, 13)


def main():
    _test()


if __name__ == '__main__':
    main()
