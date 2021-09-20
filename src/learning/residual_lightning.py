import argparse
import datetime
# import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import random
import torch
from torch.utils.data import DataLoader

from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
# from src.drone_models.residual.groundthruth import GroundTruth  # Sanity check
# from src.drone_models.residual.mlp import MLPResidual
# from src.drone_models.residual.cnn_lightning import CNNResidual
from src.drone_models.residual.gru_cnn import GRUCNNResidual
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist, is_cluster

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_residual_training/"
mkdir_if_not_exist([ANALYSIS_PATH])

GPU_COUNT = int(is_cluster())
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

print(f"GPU_COUNT: {GPU_COUNT}")


def main():
    # Args are hyper parameters to optimize
    parser = argparse.ArgumentParser(description="Identify model on data.")
    parser.add_argument('--batch_size', default=128, type=int, help="Size of a batch")
    # parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate")
    # parser.add_argument('--horizon', default=1, type=int, help="Number of time step to predict by rollout")
    # parser.add_argument('--past', default=60, type=int, help="Number of past information to take in the model")
    # parser.add_argument('--unit_size', default=50, type=int, help="Number of unit for the hidden layers")
    # parser.add_argument('--activation', default="tanh", type=str, help="The activation function to use")
    # parser.add_argument('--layer', default=3, type=int, help="Number of layer in the neural network")
    # parser.add_argument('--embedding', default="depth", type=str, help="Type of embedding to use")
    parser = GRUCNNResidual.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)  # accelerator, max_epochs
    args = parser.parse_args()

    # Training parameters
    activation = None
    if args.activation == "tanh":
        activation = torch.tanh  # torch.tanh or f.relu
    elif args.activation == "relu":
        activation = torch.relu  # torch.tanh or f.relu

    # Tensorboard
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/training/
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
               f"{activation.__name__}_"
               f"l{args.layer}_"
               f"{args.embedding}"
               )
    training_path = ANALYSIS_PATH + f"{GRUCNNResidual.__name__}_{args.embedding}/"

    # Reproducibility
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tb_logger = pl_loggers.TensorBoardLogger(training_path, version=version)
    analytic_model = DynamicalModel(param="identification", dtype=torch.float32)
    run = GRUCNNResidual(
        activation=activation,
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
                            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=bool(GPU_COUNT))

    validation_data = DataLoader(EurocMAVDatasetHorizon("validation",
                                                        traj_len=args.past + args.horizon,
                                                        dtype=torch.float32,
                                                        batch_dim=False),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=bool(GPU_COUNT))

    if is_cluster():
        trainer = pl.Trainer(gpus=GPU_COUNT, logger=tb_logger, max_epochs=args.max_epochs, accelerator="ddp")
    else:
        trainer = pl.Trainer(gpus=GPU_COUNT, logger=tb_logger, max_epochs=args.max_epochs)
    trainer.fit(run, train_data, validation_data)


if __name__ == "__main__":
    main()
