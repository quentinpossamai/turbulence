import argparse
import datetime
# import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
import random
import torch
from torch.utils.data import DataLoader

from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
# from src.drone_models.residual.groundthruth import GroundTruth  # Sanity check
# from src.drone_models.residual.mlp import MLPResidual
# from src.drone_models.residual.cnn_lightning import CNNResidual
from src.drone_models.residual.gru_cnn import GRUCNNResidual
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

EUROC_MAV_RESIDUAL = GENERAL_ANALYSIS_PATH + "euroc_mav_residual_training/"
mkdir_if_not_exist([EUROC_MAV_RESIDUAL])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
GPU_COUNT = int(device != torch.device("cpu"))

print(f"{device=}")
print(f"{GPU_COUNT=}")


def main():
    # Args are hyper parameters to optimize
    parser = argparse.ArgumentParser(description="Identify model on data.")
    # Program args
    parser.add_argument('--batch_size', default=64, type=int, help="Size of a batch")
    # Debug
    file = "/Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/GRUCNNResidual_None/default/"
    file += "20210927_091013_GRUCNNResidual_bs64_epch10000_lr0.0001_h1_p60_us50_tanh_l3_None/"
    file += "checkpoints/epoch=358-step=210373.ckpt"

    parser.add_argument('--load_weights', default=file, type=str, help="Path to previous weights")
    # Module args
    parser = GRUCNNResidual.add_model_specific_args(parser)
    # Trainer args
    parser = pl.Trainer.add_argparse_args(parser)  # accelerator, max_epochs
    args = parser.parse_args()

    # Tensorboard
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version = (f"{now}_"
               f"{GRUCNNResidual.__name__}_"
               f"bs{args.batch_size}_"
               f"epch{args.max_epochs}_"
               f"lr{args.lr}_"
               f"h{args.horizon}_"
               f"p{args.past}_"
               f"us{args.unit_size}_"
               f"{args.activation}_"
               f"l{args.layer}_"
               f"{args.embedding}"
               )
    training_path = EUROC_MAV_RESIDUAL + f"{GRUCNNResidual.__name__}_{args.embedding}/"

    print(f"{training_path=}")
    print(f"{version=}")

    # Reproducibility
    seed = 0
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    seed_everything(seed=seed, workers=True)

    tb_logger = pl_loggers.TensorBoardLogger(training_path, version=version)
    analytic_model = DynamicalModel(param="identification", dtype=torch.float32)
    if args.load_weights is None:
        run = GRUCNNResidual(
            activation=args.activation,
            embedding=args.embedding,
            layer=args.layer,
            learning_rate=args.lr,
            model=analytic_model,
            motor_size=6,
            past=args.past,
            unit_size=args.unit_size,
        )
    else:
        print(f"Reloading weights from: {args.load_weights}")
        run = GRUCNNResidual.load_from_checkpoint(
            args.load_weights,
            activation=args.activation,
            embedding=args.embedding,
            layer=args.layer,
            learning_rate=args.lr,
            model=analytic_model,
            motor_size=6,
            past=args.past,
            unit_size=args.unit_size,
        )

    # Datasets
    train_data = DataLoader(EurocMAVDatasetHorizon("train", traj_len=args.past + args.horizon,
                                                   dtype=torch.float32,
                                                   batch_dim=False),
                            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=bool(GPU_COUNT),
                            drop_last=False)

    validation_data = DataLoader(EurocMAVDatasetHorizon("validation",
                                                        traj_len=args.past + args.horizon,
                                                        dtype=torch.float32,
                                                        batch_dim=False),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=bool(GPU_COUNT),
                                 drop_last=False)

    trainer = pl.Trainer(gpus=GPU_COUNT, logger=tb_logger,
                         # deterministic=True,
                         # check_val_every_n_epoch=1,
                         # val_check_interval=1.0
                         )
    trainer.fit(run, train_data, validation_data)


if __name__ == "__main__":
    main()
