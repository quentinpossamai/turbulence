import datetime
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.drone_models.residual.mlp import MLPResidual
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_residual_training/"
mkdir_if_not_exist([ANALYSIS_PATH])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def loss(residual_gt, residual_hat):
    mse_loss = nn.MSELoss()
    fa_loss = mse_loss(residual_gt[:, 0, :3], residual_hat[:, :3])
    tau_a_loss = mse_loss(residual_gt[:, 0, 3:], residual_hat[:, 3:])
    return fa_loss + tau_a_loss


def validate(model, dataloader, horizon, epoch):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation: {epoch}"):
            state = batch["state"][:, :horizon, :].to(device)
            motor = batch["motor"][:, :horizon, :].to(device)
            t = batch["t"].to(device)
            residual_gt = batch["residual"][:, -1:, :].to(device)

            residual_hat = model(state, motor, t[:, 1:, :] - t[:, :-1, :])
            validation_loss = loss(residual_gt=residual_gt, residual_hat=residual_hat)
            loss_log.append(validation_loss)
    return torch.mean(torch.tensor(loss_log).float())


def euroc_mav_fa():
    # Reproducibility
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Training parameters
    batch_size = 32
    epochs = 100
    memory = 100000
    horizon = 10

    model = MLPResidual(motor_size=6, n_past=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Tensorboard
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{now}_{model.__class__.__name__}_bs_{batch_size}"
    training_path = ANALYSIS_PATH + f"training/{name}/"
    print("Tensorboard logdir: ", training_path)
    writer = SummaryWriter(training_path)

    # Datasets
    train_data = DataLoader(EurocMAVDatasetHorizon("train", horizon=horizon), batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(EurocMAVDatasetHorizon("validation", horizon=horizon),
                                 batch_size=batch_size, shuffle=True)
    # num_workers=1, pin_memory=True

    global_iteration = 0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            state = batch["state"][:, :horizon, :].to(device)
            motor = batch["motor"][:, :horizon, :].to(device)
            t = batch["t"].to(device)
            residual_gt = batch["residual"][:, -1:, :].to(device)

            residual_hat = model(state, motor, t[:, 1:, :] - t[:, :-1, :])
            training_loss = loss(residual_gt=residual_gt, residual_hat=residual_hat)

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
    euroc_mav_fa()
