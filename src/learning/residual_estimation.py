import argparse
import datetime
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.drone_models.residual.groundthruth import GroundTruth
from src.drone_models.residual.mlp import MLPResidual
from src.drone_models.residual.cnn import CNNResidual
from src.drone_models.residual.gru import GRUResidual
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_residual_training/"
mkdir_if_not_exist([ANALYSIS_PATH])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

DT = torch.FloatTensor([[1 / 200]])


def loss(residual_gt, residual_hat):
    # mse_loss = nn.MSELoss()
    # fa_loss = mse_loss(residual_gt[:, :3], residual_hat[:, :3])
    # tau_a_loss = mse_loss(residual_gt[:, 3:], residual_hat[:, 3:])
    # torch.mean((residual_gt - residual_hat) ** 2, dim=1).view(-1, 1)
    return (residual_gt - residual_hat) ** 2


def validate(model, dataloader, horizon, past, epoch):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation: {epoch}"):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            residual_gt = batch["residual"][:, -1].to(device)

            # Horizon forward
            past_states = state[:, -1 - past:-1]
            past_motor = motor[:, -1 - past:-1]
            # next_state = state[:, -1:]
            residual_hat = model(past_motor, past_states, DT)

            validation_loss = loss(residual_gt=residual_gt, residual_hat=residual_hat)
            loss_log.append(validation_loss)
    return torch.cat(loss_log).mean().item()


def fit_residual(batch_size, epochs, lr, horizon, past, unit_size):
    # Reproducibility
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Training parameters
    memory = 100000

    model = CNNResidual(motor_size=6, n_past=past, unit_size=unit_size).to(device)
    # model = GRUResidual(motor_size=6, n_past=past, unit_size=unit_size).to(device)
    # model = GroundTruth().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tensorboard
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{now}_{model.__class__.__name__}_bs{batch_size}_epch{epochs}_lr{lr}_h{horizon}_p{past}_us{unit_size}"
    training_path = ANALYSIS_PATH + f"training/{name}/"
    print("Tensorboard logdir: ", training_path)
    writer = SummaryWriter(training_path)

    # Datasets
    train_data = DataLoader(EurocMAVDatasetHorizon("train", traj_len=past + horizon),
                            batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(EurocMAVDatasetHorizon("validation", traj_len=past + horizon),
                                 batch_size=batch_size, shuffle=True)
    # num_workers=1, pin_memory=True

    global_iteration = 0
    validation_loss = validate(model, validation_data, horizon, past, epoch=-1)
    writer.add_scalars("Loss", {"Validation loss": validation_loss}, global_step=global_iteration)
    writer.add_scalars("Epoch loss", {"Validation loss": validation_loss}, global_step=-1)
    training_log = []
    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            model.train()
            # Look in past states
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            residual_gt = batch["residual"][:, -1].to(device)

            # Horizon forward
            past_states = state[:, -1 - past:-1]
            past_motor = motor[:, -1 - past:-1]
            # next_state = state[:, -1:]
            residual_hat = model(past_motor, past_states, DT)

            training_loss = loss(residual_gt=residual_gt, residual_hat=residual_hat)
            reduced_training_loss = training_loss.mean()
            writer.add_scalars("Loss", {"Training loss": reduced_training_loss.item()}, global_step=global_iteration)
            training_log.append(training_loss.detach())

            optimizer.zero_grad()
            reduced_training_loss.backward()
            optimizer.step()
            global_iteration = global_iteration + 1
        writer.add_scalars("Epoch loss", {"Training loss": torch.cat(training_log).mean().item()}, global_step=epoch)

        validation_loss = validate(model, validation_data, horizon, past, epoch)
        writer.add_scalars("Loss", {"Validation loss": validation_loss}, global_step=global_iteration)
        writer.add_scalars("Epoch loss", {"Validation loss": validation_loss}, global_step=epoch)
        if validation_loss < memory:
            memory = validation_loss
            torch.save(model.state_dict(), training_path + f"{global_iteration}_weights.pt")


def main():
    parser = argparse.ArgumentParser(description="Identify model on data.")
    parser.add_argument('--batch_size', default=32, type=int, help="Size of a batch")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--past', default=20, type=int, help="Number of past information to take in the model")
    parser.add_argument('--horizon', default=1, type=int, help="Number of time step to predict by rollout")
    parser.add_argument('--unit_size', default=256, type=int, help="Number of unit for the hidden layers")
    args = parser.parse_args()
    fit_residual(args.batch_size, args.epochs, args.lr, args.horizon, args.past, args.unit_size)


if __name__ == "__main__":
    main()
