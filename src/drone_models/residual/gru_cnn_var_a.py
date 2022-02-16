import argparse
import datetime
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union, Callable, Tuple

# from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.data.habitat.loading import HabitatResidualDataset, MP3D_SCENE_LIST
from src.drone_models.model_structure import Run
from src.drone_models.analytical_models.quadrotor_4parms import NumpyDynamicalModel
import src.utils.layer as layer
from src.utils.paths import mkdir_if_not_exist, ANALYSIS_PATH


class VariantA(pl.LightningModule, Run):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariantA")
        parser.add_argument('--activation', default="tanh", type=str, help="The activation function to use")
        parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
        parser.add_argument('--past', default=40, type=int, help="Number of past information to take in the model")
        return parent_parser

    def __init__(self,
                 activation: str,
                 learning_rate: float,
                 uav_model,
                 motor_size: int,
                 past: int):
        """
        Model to predict residual f_a = f_aero. Takes motor, state and image from past timesteps.

        param activation: The activation function to use.

        param learning_rate: Learning rate for Adam optimizer.

        param uav_model: The model to use the residual model with for the rollout.

        param motor_size: The number of motors.

        param past: The number of past state and motor measures to use.
        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu

        self.learning_rate = learning_rate
        self.uav_model = uav_model
        self.motor_size = motor_size
        self.past = past
        self.save_hyperparameters(ignore=['uav_model'])

        self.depth_input_size = [1, 480, 640]
        self.feature_size = self.motor_size + 13

        # - sm (state, motor) branch
        # For the simulation past motor and state is useless
        self.sm_input_norm = nn.BatchNorm1d(self.feature_size)
        self.sm_input_drop = nn.Dropout(p=0.2)
        self.sm_fc_out = layer.FullyConnected(self.activation, dropout_p=0.5, in_features=self.feature_size,
                                              out_features=128)

        # - depth (encoding depth) branch
        self.depth_input_norm = nn.BatchNorm2d(1)  # Nbatch, Channel, Height, Width
        # self.depth_input_drop = nn.Dropout2d(0.2)  # no dropout2d for input because dropout2d clears entire channel
        # and input channel number == 1

        self.depth_conv2d_0 = layer.Conv2D(self.activation, 0.5, True, 1, 16, (7, 7), 1, 3)
        self.depth_pool_0 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_1 = layer.Conv2D(self.activation, 0.5, True, 16, 32, (3, 3), 1)
        self.depth_pool_1 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_2 = layer.Conv2D(self.activation, 0.5, True, 32, 64, (3, 3), 2)
        self.depth_pool_2 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_3 = layer.Conv2D(self.activation, 0.5, True, 64, 64, (3, 3), 1)
        self.depth_pool_3 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_4 = layer.Conv2D(self.activation, 0.5, True, 64, 32, (3, 3), 2)
        self.depth_pool_4 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_5 = layer.Conv2D(self.activation, 0.5, True, 32, 4, (3, 3), 1)
        self.depth_pool_5 = nn.MaxPool2d((2, 2))
        self.depth_emb_linear = layer.FullyConnected(None, dropout_p=0.5, in_features=4 * 13 * 18, out_features=128)
        # Time reduction
        self.depth_gru = nn.GRU(input_size=128, hidden_size=128, num_layers=3, batch_first=True)
        self.depth_gru_norm = nn.BatchNorm1d(128)
        self.depth_gru_dropout = nn.Dropout(p=0.5)

        # - reg (regression) branch
        self.reg_fc_0 = layer.FullyConnected(self.activation, 0.5, 128, 50)
        self.reg_fc_1 = layer.FullyConnected(self.activation, 0.5, 50, 50)
        self.reg_linear_out = nn.Linear(in_features=50, out_features=1)

        self.example_input_array = [torch.zeros(2, self.motor_size),
                                    torch.zeros(2, 13),
                                    torch.zeros(2, self.past, 480, 640)]

    def configure_optimizers(self):
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
                next_state = self.uav_model(motor=motor[:, i + 1],
                                            state=prediction[-1].view(-1, 13),
                                            dt=dt[:, i + 1],
                                            residual=residual).view(-1, 1, 13)
                prediction.append(next_state)
                past_state = torch.cat([past_state[:, 1:], next_state], dim=1)

        prediction = torch.cat(prediction, dim=1)[:, 1:]  # First is not a prediction
        error = ((prediction - next_state_gt) ** 2)
        return error.mean(), error.reshape(pred_shape), prediction.reshape(pred_shape)


ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING = f'{ANALYSIS_PATH}euroc_mav_residual_training/'
ANALYSIS_HABITAT_RESIDUAL_TRAINING = f'{ANALYSIS_PATH}habitat_residual_training/'

SAVING_PATH = ANALYSIS_HABITAT_RESIDUAL_TRAINING
mkdir_if_not_exist([SAVING_PATH])

use_cuda_if_possible = True
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')
GPU_COUNT = int(device != torch.device('cpu'))

print(f'device={device}')
print(f'GPU_COUNT={GPU_COUNT}')


def main():
    # TODO update main from gru_cnn_var_d.py
    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Identify model on data.')
    parser.add_argument('--batch_size', default=2, type=int, help='Size of a batch')
    parser.add_argument('--load_weights', default=None, type=str, help='Path to previous weights')
    residual_model_class = VariantA
    parser = residual_model_class.add_model_specific_args(parser)  # Module args
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args accelerator, max_epochs
    args = parser.parse_args()

    # Display entire dataset
    for name in ['train', 'valid', 'test']:
        dataset_stats = HabitatResidualDataset(name, args.past, load_depth=False, load_depth_emb=False,
                                               load_norm_emb=False)
        print(name)
        for scene_number, scene_traj in dataset_stats.index.items():
            print(f'{scene_number} - {MP3D_SCENE_LIST[scene_number]}: {scene_traj}')
        print('\n\n')
    print(f'Data length: | {len(dataset_stats)}')

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    motor_size = 4
    h_prms = {'time_start': now,  # hyperparameters
              'name': residual_model_class.__name__,
              'activation': args.activation,
              'batch_size': args.batch_size,
              'lr': args.lr,
              'motor_size': motor_size,
              'max_epochs': args.max_epochs,
              'past': args.past}
    name = residual_model_class.__name__
    version = f'{now}'
    training_path = f'{SAVING_PATH}/{name}/{version}/'
    mkdir_if_not_exist([training_path])
    pickle.dump(h_prms, open(f'{training_path}h_prms.pkl', 'wb'))
    print(f'training_path={training_path}')
    print(f'name={name} | version={version}')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=SAVING_PATH, name=name, version=version, log_graph=True,
                                             default_hp_metric=True)
    tb_logger.experiment.add_hparams(h_prms, {})

    # Reproducibility
    seed = 0
    seed_everything(seed=seed, workers=True)

    # Drone model for rollout
    drone_width, drone_height = 0.1, 0.025
    volumetric_mass = 200
    g = 9.81
    m = volumetric_mass * drone_width ** 2 * drone_height
    drone_prms = {'width': drone_width,  # [m]
                  'height': drone_height,  # [m]
                  'm': m,  # [kg]
                  'kt': m * g / (4 * 150 ** 2),  #
                  'km': m * g / (4 * 150 ** 2) * 2 ** 0.5 * drone_width / 2,  #
                  'l': 2 ** 0.5 * drone_width / 2,  # [m]
                  'ix': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iy': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iz': 1 / 3 * (drone_width ** 2 + drone_width ** 2) * m,  # [kg.m2]
                  'prop_radius': drone_width * 0.1}  # [m]
    theta = np.array([drone_prms['kt'] / drone_prms['m'],
                      drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                      drone_prms['km'] / drone_prms['iz'],
                      drone_prms['iz'] / drone_prms['ix']], dtype=np.float32)
    quadrotor = NumpyDynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=np.float32,
                                    g=g)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=training_path, version=version, log_graph=True)
    if args.load_weights is None:
        run = residual_model_class(activation=args.activation,
                                   learning_rate=args.lr,
                                   motor_size=motor_size,
                                   past=args.past,
                                   uav_model=quadrotor)
    else:
        print(f'\nReloading weights from: {args.load_weights}')
        run = residual_model_class.load_from_checkpoint(args.load_weights,
                                                        activation=args.activation,
                                                        learning_rate=args.lr,
                                                        motor_size=motor_size,
                                                        past=args.past,
                                                        uav_model=quadrotor)
    num_workers = 12 if GPU_COUNT else 8
    # Datasets
    train_data = DataLoader(HabitatResidualDataset('train', past=args.past, load_depth=True,
                                                   load_depth_emb=False,
                                                   load_norm_emb=False),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

    validation_data = DataLoader(HabitatResidualDataset('valid', past=args.past, load_depth=True,
                                                        load_depth_emb=False,
                                                        load_norm_emb=False),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    trainer = pl.Trainer(gpus=GPU_COUNT,
                         logger=tb_logger,
                         deterministic=True,
                         max_epochs=args.max_epochs,
                         # fast_dev_run=False,
                         # track_grad_norm=2,
                         # accelerator=None,
                         log_every_n_steps=50,  # default  50
                         accelerator='ddp' if GPU_COUNT else None,
                         log_gpu_memory='True')
    trainer.fit(run, train_data, validation_data)


if __name__ == "__main__":
    main()
