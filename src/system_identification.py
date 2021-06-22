import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.blackbird.loading import BlackbirdDatasetHorizon
from src.drone_models.learned_sabatino import Quadrotor
from src.drone_models.mlp_quadrotor import MLPQuadrotor
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
mkdir_if_not_exist([ANALYSIS_PATH])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def loss(next_state_gt, next_state_hat):
    mse_loss = nn.MSELoss()

    position_loss = mse_loss(next_state_hat[:, :, :3], next_state_gt[:, :, :3])
    quat_loss = mse_loss(next_state_hat[:, :, 3:7], next_state_gt[:, :, 3:7])
    linear_speed_loss = mse_loss(next_state_hat[:, :, 7:10], next_state_gt[:, :, 7:10])
    angular_velocity_loss = mse_loss(next_state_hat[:, :, 10:], next_state_gt[:, :, 10:])

    return position_loss + quat_loss + linear_speed_loss + angular_velocity_loss
    # return 1/10 * position_loss + 10 * quat_loss + linear_speed_loss + angular_velocity_loss
    # return linear_speed_loss + angular_velocity_loss


def validate(model, dataloader, horizon, epoch):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation: {epoch}"):
            state = batch["state"].to(device).transpose(0, 1)
            motor = batch["motor"].to(device).transpose(0, 1)
            t = batch["t"].to(device).transpose(0, 1)
            state_hat = [state[0, :, :]]
            for j in range(horizon):
                state_hat.append(model(state_hat[-1], motor[j], t[j+1] - t[j]))
            state_hat = torch.stack(state_hat)
            validation_loss = loss(next_state_gt=state, next_state_hat=state_hat)
            loss_log.append(validation_loss)
    return torch.mean(torch.tensor(loss_log).float())


def si_blackbird_quadrotor():
    # Reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Training parameters
    batch_size = 32
    epochs = 100
    memory = 100000
    horizon = 100

    model = MLPQuadrotor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load previous weights
    weights = torch.load(ANALYSIS_PATH + "training/20210621_111255_MLPQuadrotor_bs_64000/2204_weights.pt")
    model.load_state_dict(weights)

    # Tensorboard
    # /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/tensorboard/
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{now}_{model.__class__.__name__}_bs_{batch_size}"
    training_path = ANALYSIS_PATH + f"training/{name}/"
    print("Tensorboard logdir: ", training_path)
    # writer = SummaryWriter(training_path)

    # Datasets
    train_data = DataLoader(BlackbirdDatasetHorizon("train", horizon=horizon), batch_size=batch_size, shuffle=True)
    validation_dataset = BlackbirdDatasetHorizon("validation", horizon=horizon)
    validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    # num_workers=1, pin_memory=True

    global_iteration = 0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            state = batch["state"].to(device).transpose(0, 1)
            motor = batch["motor"].to(device).transpose(0, 1)
            t = batch["t"].to(device).transpose(0, 1)

            state_hat = [state[0, :, :]]
            for j in range(horizon):
                state_hat.append(model(state_hat[-1], motor[j], t[j+1] - t[j]))
            state_hat = torch.stack(state_hat)
            training_loss = loss(next_state_gt=state, next_state_hat=state_hat)

            writer.add_scalars("Loss", {"Training loss": training_loss.item()}, global_step=global_iteration)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            global_iteration = global_iteration + 1

        validation_loss = validate(model, validation_data, horizon, epoch)
        writer.add_scalars("Loss", {"Validation loss": validation_loss.item()}, global_step=global_iteration)
        if validation_loss < memory:
            memory = validation_loss
            torch.save(model.state_dict(), training_path + f"{global_iteration}_weights.pt")


if __name__ == "__main__":
    si_blackbird_quadrotor()
