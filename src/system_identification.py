import datetime
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.blackbird.loading import BlackbirdDataset, load_flight, TORCH_DATASET_PATH
from src.drone_models.learned_sabatino import Quadrotor
from src.drone_models.mlp_quadrotor import MLPQuadrotor
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
mkdir_if_not_exist([ANALYSIS_PATH])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def loss(state_gt, dstate_hat, next_state_gt, dt):
    mse_loss = nn.MSELoss()
    next_state_hat = state_gt + dstate_hat * dt

    # position_loss = mse_loss(next_state_hat[:, :3], next_state_gt[:, :3])
    # quat_loss = mse_loss(next_state_hat[:, 3:7], next_state_gt[:, 3:7])
    linear_speed_loss = mse_loss(next_state_hat[:, 7:10], next_state_gt[:, 7:10])
    angular_velocity_loss = mse_loss(next_state_hat[:, 10:], next_state_gt[:, 10:])

    # return position_loss + quat_loss + linear_speed_loss + angular_velocity_loss
    # return 1/10 * position_loss + 10 * quat_loss + linear_speed_loss + angular_velocity_loss
    return linear_speed_loss + angular_velocity_loss


def validate(model, dataloader, global_iteration, epoch):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation: {epoch}"):
            # time_tracker = tqdm(dataloader, desc="Validation")
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            next_state = batch["next_state"].to(device)
            dt = batch["next_t"].to(device) - batch["t"].to(device)
            output = model(state, motor)  # dstate
            validation_loss = loss(state_gt=state, dstate_hat=output, next_state_gt=next_state, dt=dt)
            loss_log.append(validation_loss)
        # time_tracker.update(1)
    return torch.mean(torch.tensor(loss_log).float())


def si_blackbird_quadrotor():
    # Training parameters
    batch_size = 64000
    epochs = 100
    memory = 100000

    model = MLPQuadrotor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    weights = torch.load(ANALYSIS_PATH + "training/20210618_153656_MLPQuadrotor_bs_64000/1160_weights.pt")
    model.load_state_dict(weights)

    # Tensorboard
    # /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/tensorboard/
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{now}_{model.__class__.__name__}_bs_{batch_size}"
    training_path = ANALYSIS_PATH + f"training/{name}/"
    print("Tensorboard logdir: ", training_path)
    writer = SummaryWriter(training_path)

    # Datasets
    train_dataset = BlackbirdDataset("train")
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = BlackbirdDataset("validation")
    validation_data = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True)
    # num_workers=1, pin_memory=True

    global_iteration = 0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            next_state = batch["next_state"].to(device)
            dt = batch["next_t"].to(device) - batch["t"].to(device)
            output = model(state, motor)  # dstate
            training_loss = loss(state_gt=state, dstate_hat=output, next_state_gt=next_state, dt=dt)

            writer.add_scalars("Loss", {"Training loss": training_loss.item()}, global_step=global_iteration)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            global_iteration = global_iteration + 1

        validation_loss = validate(model, validation_data, global_iteration, epoch)
        writer.add_scalars("Loss", {"Validation loss": validation_loss.item()}, global_step=global_iteration)
        if validation_loss < memory:
            memory = validation_loss
            torch.save(model.state_dict(), training_path + f"{global_iteration}_weights.pt")


if __name__ == "__main__":
    si_blackbird_quadrotor()
