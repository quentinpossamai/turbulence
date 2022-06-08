import argparse
import datetime
import numpy as np
import pickle
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union, Callable, Tuple, Dict
import wandb

# from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
import src.data.habitat.loading as habitat
from src.utils.profiler import profiler
from src.drone_model.quadrotor_4parms import DynamicalModel
from src.utils.path import mkdir, ANALYSIS_PATH


class AllHeightNoGRU(pl.LightningModule):
    def __init__(self,
                 activation: str,
                 l2: float,
                 learning_rate: float,
                 uav_model: object,
                 motor_size: int,
                 past: int):
        """
        Model to predict residual f_a = f_aero. Takes motor, state and image from past timesteps.

        param activation: The activation function to use.

        param l2: float to penalize weights with l2 loss.

        param learning_rate: Learning rate for Adam optimizer.

        param uav_model: Model of drone to use for rollout.

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
        self.l2 = l2
        self.learning_rate = learning_rate
        self.motor_size = motor_size
        self.past = past

        self.uav_model = uav_model
        self.save_hyperparameters('activation', 'l2', 'learning_rate', 'motor_size', 'past',
                                  ignore=['uav_model'], logger=True)
        self.depth_size = [480, 640]
        self.feature_size = self.motor_size + 13

        # - emb (embedding) branch
        # self.emb_input_norm = nn.BatchNorm1d(self.emb_input_size)  # N, past, 128 or 256
        # self.emb_input_drop = nn.Dropout(0.)
        self.depth_conv2d_0 = nn.Conv2d(1, 16, (7, 7), 1, 3)
        self.depth_pool_0 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_1 = nn.Conv2d(16, 32, (3, 3), 1)
        self.depth_conv2d_2 = nn.Conv2d(32, 64, (3, 3), 2)
        self.depth_conv2d_3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.depth_pool_3 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_4 = nn.Conv2d(64, 32, (3, 3), 2)
        self.depth_pool_4 = nn.MaxPool2d((2, 2))
        self.depth_conv2d_5 = nn.Conv2d(32, 4, (3, 3), 1)
        self.depth_pool_5 = nn.MaxPool2d((2, 2))
        self.depth_linear_0 = nn.Linear(in_features=4 * 6 * 8, out_features=128)
        self.depth_linear_1 = nn.Linear(128, 50)

        # Time reduction
        # self.emb_gru = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        # self.emb_gru_norm = nn.BatchNorm1d(50)
        # self.emb_gru_dropout = nn.Dropout(p=0.)

        # - reg (regression) branch
        self.reg_linear_0 = nn.Linear(50, 50)
        self.reg_linear_1 = nn.Linear(50, 50)
        self.reg_linear_out = nn.Linear(in_features=50, out_features=1)

        self.example_input_array = [torch.zeros(2, self.past, 480, 640)]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2)

    def forward(self, depth: torch.Tensor):
        r"""
        Estimate the residual

        param depth: (*, past, 128 or 256)

        Generally * means batch dimension or more but must exist during training to compute batch_norm.

        return: (*, value=1)
        """
        # - Input processing
        assert list(depth.shape[-3:]) == [self.past, 480, 640]

        # - emb branch
        out = self.activation(self.depth_conv2d_0(depth.reshape((-1, 1, 480, 640))))
        out = self.depth_pool_0(out)
        out = self.activation(self.depth_conv2d_1(out))
        out = self.activation(self.depth_conv2d_2(out))
        out = self.activation(self.depth_conv2d_3(out))
        out = self.depth_pool_3(out)
        out = self.activation(self.depth_conv2d_4(out))
        out = self.depth_pool_4(out)
        out = self.activation(self.depth_conv2d_5(out))
        out = self.depth_pool_5(out)
        out = self.activation(self.depth_linear_0(out.reshape((-1, 4 * 6 * 8))))
        out = self.activation(self.depth_linear_1(out))

        # out, _ = self.emb_gru(out.reshape((-1, self.past, 50)))  # gru output with shape: (*, step, feature_size)
        # out = out[:, -1, :]  # last hidden states, (*, feature_size)
        emb_out = out  # (batch, feature_size)

        # - reg branch
        # out = self.reg_fc_0(sm_out + emb_out)
        out = self.activation(self.reg_linear_0(emb_out))
        out = self.activation(self.reg_linear_1(out))
        out = self.reg_linear_out(out)
        return out

    def step(self, batch):
        """
        param batch: {motor: (batch, 6),
                      state: (batch, 13),
                      depth: (batch, self.past, 480, 640),
                      residual: (batch, 1),
                      surface_dist: (batch, 1)}

        return: Loss
        """
        z_hat = self.forward(depth=batch['depth'])
        z_gt = batch["surface_dist"]

        loss = (z_gt - z_hat) ** 2
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


ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING = f'{ANALYSIS_PATH}euroc_mav_residual_training/'
ANALYSIS_HABITAT_RESIDUAL_TRAINING = f'{ANALYSIS_PATH}habitat_residual_training/'

SAVING_PATH = ANALYSIS_HABITAT_RESIDUAL_TRAINING
mkdir([SAVING_PATH])


def main():
    profiler()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    version = f'{now}'
    residual_model_class = AllHeightNoGRU
    name = residual_model_class.__name__
    training_path = f'{SAVING_PATH}{name}/{version}/'
    mkdir([training_path])

    identifier = f'{name}-{version}-debug'
    identifier = identifier if torch.cuda.is_available() else identifier + '_local'
    wandb.init(name=identifier,
               id=identifier,
               project='turbulence_residual',
               dir=training_path)
    logger = WandbLogger(experiment=wandb.run)

    use_cuda_if_possible = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')
    gpu_count = int(device != torch.device('cpu'))

    print(f'device={device}')
    print(f'GPU_COUNT={gpu_count}')

    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Identify model on data.')
    parser.add_argument('--activation', default="relu", type=str, help="The activation function to use")
    parser.add_argument('--batch_size', default=20, type=int, help='Size of a batch')
    parser.add_argument('--l2', default=1e-4, type=float, help="Number of past information to take in the model")
    parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
    parser.add_argument('--past', default=40, type=int, help="Number of past information to take in the model")
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args accelerator, max_epochs
    args = parser.parse_args()

    # Display entire dataset
    for name in ['train', 'valid', 'test']:
        dataset_stats = habitat.HabitatResidualDataset(name, args.past, load_depth=False, load_depth_emb=False,
                                                       load_norm_emb=False, normed_depth=False)
        print(name)
        for scene_number, scene_traj in dataset_stats.index.items():
            print(f'{scene_number} - {habitat.MP3D_SCENE_LIST[scene_number]}: {scene_traj}')
        print(f'Data length: | {len(dataset_stats)}')
        print('\n\n')

    # Analysis parameters
    motor_size = 4
    h_prms = {'time_start': now,  # hyperparameters
              'name': residual_model_class.__name__,
              'activation': args.activation,
              'batch_size': args.batch_size,
              'l2': args.l2,
              'lr': args.lr,
              'motor_size': motor_size,
              'max_epochs': args.max_epochs,
              'past': args.past}
    pickle.dump(h_prms, open(f'{training_path}h_prms.pkl', 'wb'))
    print(f'training_path={training_path}')
    print(f'name={name} | version={version}')

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
    quadrotor = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=np.float32, g=g)

    run = residual_model_class(activation=args.activation,
                               l2=args.l2,
                               learning_rate=args.lr,
                               motor_size=motor_size,
                               past=args.past,
                               uav_model=quadrotor)
    logger.watch(run)
    # args.resume_from_checkpoint = f'{ANALYSIS_PATH}habitat_residual_training/VariantD/20220215_163515/' \
    #                               f'checkpoints/epoch=16-step=832.ckpt'
    num_workers = 12 if gpu_count else 0
    # Datasets
    load_depth = True
    load_depth_emb = False
    load_norm_emb = False
    print(f'Load depth {load_depth} | load_depth_emb {load_depth_emb} | load_norm_emb {load_norm_emb}')
    train_data = DataLoader(habitat.HabitatResidualDataset('train', past=args.past, load_depth=load_depth,
                                                           load_depth_emb=load_depth_emb,
                                                           load_norm_emb=load_norm_emb,
                                                           normed_depth=False),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

    validation_data = DataLoader(habitat.HabitatResidualDataset('valid', past=args.past, load_depth=load_depth,
                                                                load_depth_emb=load_depth_emb,
                                                                load_norm_emb=load_norm_emb,
                                                                normed_depth=False),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    checkpoint_callback = ModelCheckpoint(dirpath=training_path,
                                          filename='{epoch}-{step}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          mode='min')
    trainer = pl.Trainer(gpus=gpu_count,
                         logger=logger,
                         deterministic=True,
                         max_epochs=args.max_epochs,
                         # fast_dev_run=False,
                         # track_grad_norm=2,
                         # accelerator=None,
                         log_every_n_steps=50,  # default  50
                         accelerator='ddp' if gpu_count else None,
                         plugins=DDPPlugin(find_unused_parameters=False),
                         log_gpu_memory='False',
                         resume_from_checkpoint=args.resume_from_checkpoint,
                         callbacks=[checkpoint_callback])

    if args.resume_from_checkpoint is None:
        trainer.validate(run, dataloaders=validation_data)
    trainer.fit(run, train_data, validation_data)
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
