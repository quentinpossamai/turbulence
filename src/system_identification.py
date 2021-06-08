import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.blackbird.blackbird_loading import BlackbirdDataset
from src.drone_models.learned_quadrotor import Quadrotor
from src.utils.paths import MODELS_PATH, ANALYSIS_PATH

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def mse_loss(pred, target):
    return nn.MSELoss()(pred, target)


def validate(model, dataloader):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            dstate = batch["dstate"].to(device)
            output = model(state, motor)  # dstate
            loss = mse_loss(output[:, 3:9], dstate[:, 3:9])
            loss_log.append(loss)
    return torch.mean(torch.tensor(loss_log).float())


def si_blackbird_quadrotor():
    batch_size = 32
    epochs = 1
    memory = 100000

    model = Quadrotor().to(device)
    train_data = DataLoader(BlackbirdDataset("train"), batch_size=batch_size, shuffle=True, num_workers=1,
                            pin_memory=True)
    validation_data = DataLoader(BlackbirdDataset("validation"), batch_size=batch_size, shuffle=True, num_workers=1,
                                 pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_log = {"training_loss": [], "validation_loss": []}

    for epoch in range(epochs):
        model.train()
        print("Epoch ", epoch)
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            dstate = batch["dstate"].to(device)
            output = model(state, motor)  # dstate
            training_loss = mse_loss(output[:, 3:9], dstate[:, 3:9])  # Only the linear and angular acceleration

            training_log["training_loss"].append(training_loss)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            # if i % 10000 == 0:
            #     validation_loss = validate(model, validation_data)
            #     training_log["validation_loss"].append(validation_loss)
            #     if memory < validation_loss:
            #         memory = validation_loss
            #         torch.save(model.state_dict(), MODELS_PATH + "blackbird_learned_quadrotor.pt")

    reports_path = ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
    p = pathlib.Path(reports_path)
    p.mkdir(parents=True, exist_ok=True)
    pickle.dump(training_log, open(reports_path + "training_loss.pkl", "wb"))


if __name__ == "__main__":
    si_blackbird_quadrotor()
