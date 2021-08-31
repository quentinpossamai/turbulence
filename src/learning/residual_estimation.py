import argparse
import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
from src.drone_models.residual.groundthruth import GroundTruth  # Sanity check
from src.drone_models.residual.mlp import MLPResidual
from src.drone_models.residual.cnn import CNNResidual
from src.drone_models.residual.gru import GRUResidual
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist
from src.log import Logger

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_residual_training/"
mkdir_if_not_exist([ANALYSIS_PATH])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

DT = torch.FloatTensor([[1 / 200]])
MEMORY = 100000


def main():
    # Args are hyper parameters to optimize
    parser = argparse.ArgumentParser(description="Identify model on data.")
    parser.add_argument('--batch_size', default=128, type=int, help="Size of a batch")
    parser.add_argument('--epochs', default=10000, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--past', default=20, type=int, help="Number of past information to take in the model")
    parser.add_argument('--horizon', default=1, type=int, help="Number of time step to predict by rollout")
    parser.add_argument('--unit_size', default=50, type=int, help="Number of unit for the hidden layers")
    args = parser.parse_args()

    # Training parameters
    activation = f.relu  # f.tanh
    seed = 0

    # Tensorboard
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{now}_"
            f"{CNNResidual.__name__}_"
            f"bs{args.batch_size}_"
            f"epch{args.epochs}_"
            f"lr{args.lr}_"
            f"h{args.horizon}_"
            f"p{args.past}_"
            f"us{args.unit_size}_"
            f"{activation.__name__}")
    training_path = ANALYSIS_PATH + f"training/{name}/"
    print("Tensorboard logdir: ", training_path)

    # Reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    log = Logger(run_name=name, saving_path=training_path)
    analytic_model = DynamicalModel(paper=False, dtype=torch.float32)
    run = CNNResidual(log=log,
                      model=analytic_model,
                      past=args.past,
                      unit_size=args.unit_size,
                      activation=activation,
                      device=device)
    optimizer = run.configure_optimizers(lambda x: torch.optim.Adam(x, lr=args.lr))
    run = run.to(device)

    # Datasets
    train_data = DataLoader(EurocMAVDatasetHorizon("train", traj_len=args.past + args.horizon,
                                                   dtype=torch.float32,
                                                   for_loader=True),
                            batch_size=args.batch_size, shuffle=True)
    validation_data = DataLoader(EurocMAVDatasetHorizon("validation",
                                                        traj_len=args.past + args.horizon,
                                                        dtype=torch.float32,
                                                        for_loader=True),
                                 batch_size=args.batch_size, shuffle=True)
    # num_workers=1, pin_memory=True

    run.log.g_idx = 0
    for run.log.epoch in range(args.epochs):
        validate(run, validation_data)
        for run.log.tb_idx, batch in enumerate(tqdm(train_data, desc=f"Epoch: {run.log.epoch}")):
            training_loss = run.training_step(train_batch=batch)
            reduced_training_loss = training_loss.mean()
            run.log.train_epoch_loss_log.append(training_loss.detach())

            optimizer.zero_grad()
            reduced_training_loss.backward()
            optimizer.step()

            run.log.g_idx = run.log.g_idx + 1
        # Update logger
        run.log.sw.add_scalars("Epoch loss", {"Training loss": torch.cat(run.log.train_epoch_loss_log).mean().item()},
                               global_step=run.log.epoch)
        run.log.train_loss_log.append(run.log.train_epoch_loss_log)
        run.log.train_epoch_loss_log = []


def validate(run, dataloader):
    global MEMORY
    run.eval()
    run.log.valid_epoch_loss_log = []
    with torch.no_grad():
        for run.log.vb_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation: {run.log.epoch}")):
            batch_loss = run.validation_step(val_batch=batch)
            run.log.valid_epoch_loss_log.append(batch_loss)
    epoch_loss = torch.cat(run.log.valid_epoch_loss_log).mean().item()

    run.log.sw.add_scalars("Epoch loss", {"Valid loss": epoch_loss}, global_step=run.log.epoch)

    # Saving weights
    if epoch_loss < MEMORY:
        MEMORY = epoch_loss
        torch.save(run.state_dict(), run.log.saving_path + f"{run.log.epoch}_weights.pt")

    # Update logger
    run.log.valid_loss_log.append(run.log.valid_epoch_loss_log)
    run.log.valid_epoch_loss_log = []
    return epoch_loss


if __name__ == "__main__":
    main()
