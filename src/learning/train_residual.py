import argparse
import datetime
# import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
import pickle
import torch
from torch.utils.data import DataLoader

from src.drone_models.analytical_models.quadrotor_4parms import NumpyDynamicalModel
# from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.data.habitat.loading import HabitatDataset
from src.drone_models.residual.gru_cnn_variant_A import VariantA
from src.utils.paths import mkdir_if_not_exist, ANALYSIS_PATH

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
    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Identify model on data.')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a batch')
    parser.add_argument('--load_weights', default=None, type=str, help='Path to previous weights')
    residual_model_class = VariantA
    parser = residual_model_class.add_model_specific_args(parser)  # Module args
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args accelerator, max_epochs
    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # TODO Create a yaml to save hyperparameters and loss ? pkl ?
    motor_size = 4
    h_prms = {'time_start': now,  # hyperparameters
              'name': residual_model_class.__name__,
              'activation': args.activation,
              'batch_size': args.batch_size,
              'lr': args.lr,
              'motor_size': motor_size,
              'max_epochs': args.max_epochs,
              'past': args.past,
              'unit_size': args.unit_size}
    version = f'{now}'
    training_path = SAVING_PATH + f'{residual_model_class.__name__}/'
    mkdir_if_not_exist([training_path])
    pickle.dump(h_prms, open(f'{training_path}h_prms.pkl', 'wb'))
    print(f'training_path={training_path}')
    print(f'version={version}')

    # Reproducibility
    seed = 0
    seed_everything(seed=seed, workers=True)

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
                                   model=quadrotor,
                                   motor_size=6,
                                   past=args.past,
                                   unit_size=args.unit_size)
    else:
        print(f'\nReloading weights from: {args.load_weights}')
        run = residual_model_class.load_from_checkpoint(args.load_weights,
                                                        activation=args.activation,
                                                        learning_rate=args.lr,
                                                        model=quadrotor,
                                                        motor_size=6,
                                                        past=args.past,
                                                        unit_size=args.unit_size)

    num_workers = 12 if GPU_COUNT else 0
    # Datasets
    train_data = DataLoader(HabitatDataset('train', past=args.past),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

    validation_data = DataLoader(HabitatDataset('validation', past=args.past),
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
                         accelerator='ddp' if GPU_COUNT else None,
                         log_gpu_memory='True')
    trainer.fit(run, train_data, validation_data)


if __name__ == "__main__":
    main()
