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
import sys
import wandb

# from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
import src.data.habitat.loading as habitat
from src.drone_models.model_structure import Run
from src.utils.profiler import profiler
from src.drone_models.analytical_models.quadrotor_4parms import DynamicalModel
from src.utils.path import mkdir, ANALYSIS_PATH


class AllHeight(pl.LightningModule):
    def __init__(self, activation: str, dropout: float, l2: float, lr: float, motor_size: int,
                 past: int):
        """
        Model to predict residual f_a = f_aero. Takes motor, state and image from past timesteps.

        Args:
            activation: The activation function to use.
            dropout: The dropout probability.
            l2: Float to penalize weights with l2 loss.
            lr: Learning rate for Adam optimizer.
            uav_model: Model of drone to use for rollout.
            motor_size: The number of motors.
            past: The number of past state and motor measures to use.
        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu
        self.l2 = l2
        self.learning_rate = lr
        self.motor_size = motor_size
        self.past = past

        self.save_hyperparameters(logger=True)
        self.depth_size = [480, 640]
        self.feature_size = self.motor_size + 13

        # Dropout
        # self.dropout_input = nn.Dropout(0)
        self.dropout = nn.Dropout(dropout)

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
        self.depth_linear_0 = nn.Linear(in_features=4 * 6 * 8, out_features=128)  # 192
        self.depth_linear_1 = nn.Linear(128, 50)

        # Time reduction
        if self.past > 1:
            self.emb_gru = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
            self.emb_gru_dropout = nn.Dropout(p=0.)

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

        param depth: (*, 1, 480, 640)

        Generally * means batch dimension or more but must exist during training to compute batch_norm.

        return: (*, 1)
        """
        # - Input processing
        assert list(depth.shape[-3:]) == [self.past, 480, 640]

        # - emb branch
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

        out = self.dropout(out.reshape((-1, 4 * 6 * 8)))
        out = self.activation(self.depth_linear_0(out))
        out = self.dropout(out)
        out = self.activation(self.depth_linear_1(out))
        out = self.dropout(out)

        if self.past > 1:
            out, _ = self.emb_gru(out.reshape((-1, self.past, 50)))  # gru output with shape: (*, step, feature_size)
            out = out[:, -1, :]  # last hidden states, (*, feature_size)
        emb_out = out  # (batch, feature_size)

        # - reg branch
        # out = self.reg_fc_0(sm_out + emb_out)
        out = self.activation(self.reg_linear_0(emb_out))
        out = self.dropout(out)
        out = self.activation(self.reg_linear_1(out))
        out = self.dropout(out)
        out = self.reg_linear_out(out)
        return out

    def loss(self, z_gt, z_hat, label):
        """
        Args:
            z_gt: (batch, 1),
            z_hat: (batch, 1),
            label: (batch, self.past, 1)

        Returns: The loss used by the model.
        """
        data_to_use = 1 - torch.any(label == 1, dim=1).float()  # check for valid data,
        # data with label 1 is invalid

        weight_surface_diff = 1
        surface_diff = data_to_use * ((label == 2)[:, -1].float() * (weight_surface_diff - 1) + 1)

        weight_noisy_data = 1
        noisy_data = data_to_use * ((label == 3)[:, -1].float() * (weight_noisy_data - 1) + 1)

        loss = data_to_use * ((z_gt - z_hat) ** 2).squeeze()
        loss = loss * surface_diff
        loss = loss * noisy_data
        return loss

    def training_step(self, train_batch, batch_idx):
        """
        Args:
            train_batch: {motor: (batch, 6),
                          state: (batch, 13),
                          depth: (batch, self.past, 480, 640),
                          residual: (batch, 1),
                          label: (batch, self.past, 1)}
            batch_idx:

        Returns:

        """
        z_gt = train_batch["surface_dist"]
        z_hat = self.forward(depth=train_batch['depth'])
        label = train_batch["label"]
        training_loss = self.loss(z_gt, z_hat, label).mean()
        self.log("train_loss", training_loss)
        return training_loss

    def validation_step(self, val_batch, batch_idx):
        """
        Cf. training_step docstring.
        """
        z_gt = val_batch["surface_dist"]
        z_hat = self.forward(depth=val_batch['depth'])
        label = val_batch["label"]
        val_loss = self.loss(z_gt, z_hat, label)
        self.log("val_loss", val_loss)


ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING = f'{ANALYSIS_PATH}euroc_mav_residual_training/'
ANALYSIS_HABITAT_RESIDUAL_TRAINING = f'{ANALYSIS_PATH}habitat_residual_training/'

SAVING_PATH = ANALYSIS_HABITAT_RESIDUAL_TRAINING
mkdir([SAVING_PATH])


def main():
    profiler()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    version = f'{now}'
    residual_model_class = AllHeight
    name = residual_model_class.__name__
    training_path = f'{SAVING_PATH}{name}/{version}/'
    mkdir([training_path])

    identifier = f'{name}-{version}-raw'
    wandb.init(name=identifier,
               id=identifier,
               project='turbulence_residual',
               dir=training_path,
               mode='online' if sys.platform != 'darwin' else 'disabled')
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
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout probability')
    parser.add_argument('--l2', default=0, type=float, help="Number of past information to take in the model")
    parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
    parser.add_argument('--past', default=1, type=int, help="Number of past information to take in the model")
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args accelerator, max_epochs
    args = parser.parse_args()

    # Display entire dataset
    for sub_set_name in ['train', 'valid', 'test']:
        dataset_stats = habitat.HabitatResidualDataset(sub_set_name, args.past, load_depth=True, load_depth_emb=False,
                                                       load_norm_emb=False, normed_depth=False)
        print(sub_set_name)
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
              'dropout': args.dropout,
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

    run = residual_model_class(activation=args.activation,
                               dropout=args.dropout,
                               l2=args.l2,
                               lr=args.lr,
                               motor_size=motor_size,
                               past=args.past)
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
