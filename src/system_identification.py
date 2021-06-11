import datetime
import pathlib
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.blackbird.blackbird_loading import BlackbirdDataset
from src.drone_models.learned_quadrotor import Quadrotor
from src.utils.paths import MODELS_PATH, ANALYSIS_PATH

RESULT_FOLDER = ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def loss(state, state_gt):
    # Without the angles
    return nn.MSELoss()(state[:, 3:], state_gt[:, 3:])


def validate(model, dataloader, global_iteration, epoch):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation: {epoch}"):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            next_state = batch["next_state"].to(device)
            dt = batch["next_t"].to(device) - batch["t"].to(device)
            output = model(state, motor)  # dstate
            validation_loss = loss(output * dt + state, next_state)
            loss_log.append(validation_loss)
    return torch.mean(torch.tensor(loss_log).float())


def si_blackbird_quadrotor():
    # Training parameters
    batch_size = 64000
    epochs = 100
    memory = 100000

    # Tensorboard
    # /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/tensorboard/
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = now + f"_batch_size_{batch_size}"
    tensorboard_folder = RESULT_FOLDER + f"training/{name}/"
    print("Tensorboard logdir: ", tensorboard_folder)
    writer = SummaryWriter(tensorboard_folder)

    model = Quadrotor().to(device)
    train_data = DataLoader(BlackbirdDataset("train"), batch_size=batch_size, shuffle=True, num_workers=1,
                            pin_memory=True)
    validation_data = DataLoader(BlackbirdDataset("validation"), batch_size=batch_size, shuffle=True, num_workers=1,
                                 pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    global_iteration = 0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            next_state = batch["next_state"].to(device)
            dt = batch["next_t"].to(device) - batch["t"].to(device)
            output = model(state, motor)  # dstate
            training_loss = loss(output * dt + state, next_state)  # Only the linear and angular acceleration

            writer.add_scalars("Loss", {"Training loss": training_loss.item()}, global_step=global_iteration)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            global_iteration = global_iteration + 1

        validation_loss = validate(model, validation_data, global_iteration, epoch)
        writer.add_scalars("Loss", {"Validation loss": validation_loss.item()}, global_step=global_iteration)
        if validation_loss < memory:
            memory = validation_loss
            torch.save(model.state_dict(), RESULT_FOLDER + f"{name}.pt")

    pathlib.Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)


def _test():
    weights = torch.load(MODELS_PATH + "blackbird_learned_quadrotor/20210608_103922_batch_size_32.pt")
    print(weights)
    # OrderedDict([('a', tensor([0.0024, 0.0081, 0.0015])),
    #              ('b', tensor([[-0.0016,  0.0002,  0.0026, -0.0023],
    #                            [ 0.0009, -0.0010,  0.0002, -0.0027],
    #                            [-0.0004, -0.0020,  0.0037,  0.0024],
    #                            [-0.0025, -0.0006, -0.0015,  0.0012]]))])
    pass


if __name__ == "__main__":
    si_blackbird_quadrotor()
    # _test()
