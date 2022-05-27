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
import src.utils.path as path


class HeightEstimator(pl.LightningModule):
    def __init__(self, activation: str,
                 dropout: float,
                 l2: float,
                 lr: float,
                 motor_size: int,
                 past: int,
                 embedding_method: str):
        """
        Model to predict residual z such that f_aero(z, motor). Sequences of depth-maps from past time-steps.

        Args:
            activation: The activation function to use.
            dropout: The dropout probability of MLPs hidden layers.
            l2: Float to penalize all weights with l2 loss.
            lr: Learning rate for Adam optimizer.
            motor_size: The number of motors.
            past: The number of past depth-maps to use.
            embedding_method: The type of pre-training to use.
        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu
        self.l2 = l2
        self.lr = lr
        self.motor_size = motor_size
        self.past = past

        assert embedding_method in ['cnn', 'irs_depth', 'irs_normal']
        self.embedding_method = embedding_method

        self.save_hyperparameters(logger=True)
        embedding_size = 128

        from architectures import RegressionHead
        if embedding_method == 'cnn':
            self.depth_size = [480, 640]
            from architectures import RawEmbedding
            self.embedding_layer = RawEmbedding(embedding_size)

        if self.embedding_method == 'irs_depth':
            self.depth_size = [240, 376]
            from autoencoder_depth.Models.model_CoPhy_like import Autoencoder
            self.auto_encoder = Autoencoder(feature_size=embedding_size, in_channel=1, out_channel=1)
            self.embedding_layer = self.auto_encoder.encoder

        if self.embedding_method == 'irs_normal':
            self.depth_size = [240, 376]
            from autoencoder_depth.Models.model_CoPhy_like import Autoencoder
            self.auto_encoder = Autoencoder(feature_size=embedding_size, in_channel=1, out_channel=3)
            self.embedding_layer = self.auto_encoder.encoder

        from architectures import RegressionHead
        self.regression_head = RegressionHead(embedding_size, 50, self.past, self.dropout)

        self.example_input_array = [torch.zeros(2, self.past, self.depth_size[0], self.depth_size[1])]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

    def forward(self, depth: torch.Tensor):
        r"""
        Estimate the residual

        param depth: (*, 1, width, height)

        Generally * means batch dimension or more but must exist during training to compute batch_norm.

        return: (*, 1)
        """
        # - Input processing
        assert list(depth.shape[-3:]) == [self.past, self.depth_size[0], self.depth_size[1]]
        out = self.embedding_layer(depth)
        out = self.regression_head(out)
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
        # todo log all losses in validation to see training improvement
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
        """
        z_gt = train_batch["surface_dist"]
        z_hat = self.forward(depth=train_batch['depth'])
        label = train_batch["label"]
        training_loss = self.loss(z_gt, z_hat, label).mean()
        # self.log("train_loss", training_loss)  # todo
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


ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING = f'{path.ANALYSIS_PATH}euroc_mav_residual_training/'
ANALYSIS_HABITAT_RESIDUAL_TRAINING = f'{path.ANALYSIS_PATH}habitat_residual_training/'

SAVING_PATH = ANALYSIS_HABITAT_RESIDUAL_TRAINING
path.mkdir([SAVING_PATH])


def main():
    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Identify model on data.')
    parser.add_argument('--activation', default="relu", type=str, help="The activation function to use")
    parser.add_argument('--batch_size', default=20, type=int, help='Size of a batch')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout probability')
    parser.add_argument('--l2', default=0, type=float, help="Number of past information to take in the model")
    parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
    parser.add_argument('--past', default=1, type=int, help="Number of past information to take in the model")
    parser.add_argument('--stage', default='raw', type=str, help="The stage of the model")

    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args accelerator, max_epochs
    args = parser.parse_args()

    # ae training is already done and written elsewhere
    assert args.stage in ['cnn',
                          'irs_depth_ae', 'irs_depth_head', 'irs_depth_finetune',
                          'irs_normal_ae', 'irs_normal_head', 'irs_normal_finetune']
    # profiler()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    version = f'{now}'
    residual_model_class = HeightEstimator
    name = f'{residual_model_class.__name__}_{args.stage}'
    training_path = f'{SAVING_PATH}{name}/{version}/'
    path.mkdir([training_path])

    identifier = f'{name}_{version}'
    wandb.init(name=identifier,
               id=identifier,
               project='turbulence_residual',
               dir=training_path,
               mode='online' if sys.platform != 'darwin' else 'disabled')
    logger = WandbLogger(experiment=wandb.run)

    use_cuda_if_possible = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')

    print(f'Device used by torch: {device}')

    # Display entire dataset
    check_all_file = False if '/Users/quentin' in path.WD_PATH else True
    for split_name in ['train', 'valid', 'test']:
        dataset_stats = habitat.HabitatResidualDataset(split_name, args.past, load_depth=True, load_depth_emb=False,
                                                       load_norm_emb=False, normed_depth=False,
                                                       check_all_file=check_all_file)
        print(split_name)
        for scene_number, scene_traj in dataset_stats.index.items():
            print(f'{scene_number} - {habitat.MP3D_SCENE_LIST[scene_number]}: {scene_traj}')
        print(f'Data length: | {len(dataset_stats)}')
        print('\n\n')

    embedding_method = 'cnn'
    if 'irs_depth' in args.stage:
        embedding_method = 'irs_depth'
    if 'irs_normal' in args.stage:
        embedding_method = 'irs_normal'

    # Analysis parameters
    motor_size = 4
    h_prms = {'time_start': now,  # hyperparameters
              'name': name,
              'activation': args.activation,
              'batch_size': args.batch_size,
              'dropout': args.dropout,
              'embedding_method': embedding_method,
              'l2': args.l2,
              'lr': args.lr,
              'motor_size': motor_size,
              'max_epochs': args.max_epochs,
              'past': args.past,
              'stage': args.stage}
    pickle.dump(h_prms, open(f'{training_path}h_prms.pkl', 'wb'))
    print(f'training_path={training_path}')
    print(f'name={name} | version={version}')

    # Reproducibility
    seed = 0
    seed_everything(seed=seed, workers=True)

    run = residual_model_class(activation=h_prms['activation'],
                               dropout=h_prms['dropout'],
                               l2=h_prms['l2'],
                               lr=h_prms['lr'],
                               motor_size=h_prms['motor_size'],
                               past=h_prms['past'],
                               embedding_method=h_prms['embedding_method'])
    logger.watch(run)

    # Load previous weight depending on training stage
    if h_prms['stage'] == 'cnn':
        load_depth, load_depth_emb, load_norm_emb = True, False, False
    elif h_prms['stage'] in 'irs_depth_head':
        run.auto_encoder.load_state_dict(torch.load(f'{path.WD_PATH}models/depth_normal_autoencoder/'
                                                    f'autoencoder_depth_irs_128.nn'))
        run.auto_encoder.eval()  # Do not train the autoencoder
        load_depth, load_depth_emb, load_norm_emb = True, False, False
    elif h_prms['stage'] == 'irs_normal_head':
        run.auto_encoder.load_state_dict(torch.load(f'{path.WD_PATH}models/depth_normal_autoencoder/'
                                                    f'autoencoder_normal_irs_128.nn'))
        run.auto_encoder.eval()  # Do not train the autoencoder
    else:
        raise NotImplementedError

    # args.resume_from_checkpoint = f'{ANALYSIS_PATH}habitat_residual_training/VariantD/20220215_163515/' \
    #                               f'checkpoints/epoch=16-step=832.ckpt'
    num_workers = 12 if device == 'gpu' else 0
    # Datasets
    print(f'Load depth {load_depth} | load_depth_emb {load_depth_emb} | load_norm_emb {load_norm_emb}')
    train_data = DataLoader(habitat.HabitatResidualDataset('train', past=args.past, load_depth=load_depth,
                                                           load_depth_emb=load_depth_emb,
                                                           load_norm_emb=load_norm_emb,
                                                           normed_depth=False,
                                                           check_all_file=check_all_file),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

    validation_data = DataLoader(habitat.HabitatResidualDataset('valid', past=args.past, load_depth=load_depth,
                                                                load_depth_emb=load_depth_emb,
                                                                load_norm_emb=load_norm_emb,
                                                                normed_depth=False,
                                                                check_all_file=check_all_file),
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


def get_trained_model(name: str):
    """

    Args:
        name:
        ['none_raw',
         'none_gru',
         'depth_gru',
         'depth_gru_finetune',
         'norm_gru',
         'norm_gru_finetune']

    Returns: The strings and path associated to the model.
    """
    if name == 'none_raw':
        model_path = f'{path.ANALYSIS_PATH}habitat_residual_training/'
        model_part_path = 'AllHeight/20220513_154038/'
        model_ckpt = 'epoch=78-step=894042-val_loss=0.04.ckpt'
        model_path = f'{model_path}{model_part_path}'
        name = f'''{model_part_path.replace('/', '_')}{model_ckpt[:-5]}'''
        h_prms = pickle.load(open(f'{model_path}h_prms.pkl', 'rb'))
        model = HeightEstimator.load_from_checkpoint(f'{model_path}{model_ckpt}',
                                                     activation=h_prms['activation'],
                                                     l2=h_prms['l2'],
                                                     lr=h_prms['lr'],
                                                     uav_model=None,
                                                     motor_size=h_prms['motor_size'],
                                                     past=h_prms['past'])
    else:
        raise NotImplemented
    return model, h_prms, name


if __name__ == "__main__":
    main()
