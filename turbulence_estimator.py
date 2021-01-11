from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import util


class ModelErrorNN(nn.Module):
    def __init__(self):
        super(ModelErrorNN, self).__init__()
        self.linear1 = nn.Linear(12, 12)
        self.linear2 = nn.Linear(12, 12)
        self.linear3 = nn.Linear(12, 12)
        self.linear4 = nn.Linear(12, 12)
        self.linear5 = nn.Linear(12, 12)
        self.linear6 = nn.Linear(12, 6)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear4(x))
        x = nn.functional.relu(self.linear5(x))
        x = self.linear6(x)
        return x


def splitting_60_20_20(data: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_list, validation_list, test_list = [], [], []
    for i, df in enumerate(data):
        df.reset_index(drop=True, inplace=True)
        cut1 = np.round(len(df) * 0.6) - 1
        cut2 = cut1 + np.round(len(df) * 0.2)
        cut3 = len(df) - 1
        # assert cut3 == len(df) - 1, f"Not every samples in splits for df nº{i}. " \
        #                             f"cut3 = {cut3} | len(df) - 1: {len(df) - 1}"
        train_list.append(df.loc[:cut1])
        validation_list.append(df.loc[cut1:cut2])
        test_list.append(df.loc[cut2:cut3])

    train = pd.concat(train_list, ignore_index=True)
    validation = pd.concat(validation_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    return train, validation, test


def training():
    # Data loading
    f = util.DataFolder("euroc_mav")
    # Run in terminal: tensorboard --logdir /Users/quentin/phd/turbulence/euroc_mav/results/runs/
    log_dir = f.folders["results"][''] + "runs/"
    writer = SummaryWriter(log_dir=log_dir)
    data = []
    flight_names = {0: "euroc_mav_easy",
                    1: "euroc_mav_medium",
                    2: "euroc_mav_difficult"}
    for flight_number in range(3):
        data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "fa")
        print(f"Loading: {data_path}")
        df = pd.read_pickle(data_path).dropna()
        df["flight_name"] = flight_names[flight_number]
        data.append(df)
    print()

    # Data splitting
    train, validation, test = splitting_60_20_20(data)

    # Neural network instancing
    net = ModelErrorNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # writer.add_graph(model=net) why not working

    # Training
    # fig, ax = plt.subplots(1, 1)
    iter_counter = 0
    epochs = 10
    loop_size = range(len(train))
    progress = util.Progress(epochs * len(loop_size), "Training.", "Training done.")
    for epoch in range(epochs):
        graph_data = np.zeros((len(loop_size), 2))
        for i in loop_size:
            # get the inputs
            i1 = train["motor_speed"][i].astype(np.float32)
            i2 = train["imu0_linear_acceleration"][i].astype(np.float32)
            i3 = train["imu0_angular_velocity"][i].astype(np.float32)
            inputs = torch.from_numpy(np.hstack([i1, i2, i3])).float()
            labels = train["fa"][i].float()

            # forward pass
            outputs = net(inputs)
            outputs.retain_grad()
            loss = criterion(outputs, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss = loss.item()
            writer.add_scalars("Loss", {"Training loss": running_loss}, iter_counter)
            graph_data[i - 2, :] = (iter_counter, running_loss)
            iter_counter += 1
            progress.update()
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            pass
        # ax.plot(graph_data[:, 0], graph_data[:, 1], label=f"Epoch: {epoch}")
    # ax.set_xlabel("Number of backpropagation")
    # ax.set_ylabel("Train Loss")
    # ax.legend()
    # ax.grid()

    validation_loss_mean = []
    for i in range(len(validation)):
        # get the inputs
        i1 = validation["motor_speed"][i].astype(np.float32)
        i2 = validation["imu0_linear_acceleration"][i].astype(np.float32)
        i3 = validation["imu0_angular_velocity"][i].astype(np.float32)
        inputs = torch.from_numpy(np.hstack([i1, i2, i3])).float()
        labels = validation["fa"][i].float()

        # forward pass
        outputs = net(inputs)
        outputs.retain_grad()
        loss = criterion(outputs, labels)
        writer.add_scalars("Loss", {"Validation loss": loss.item()}, iter_counter+i+1)
        validation_loss_mean.append(loss.item())
    validation_loss_mean = np.mean(validation_loss_mean)
    print(f"Validation loss mean: {validation_loss_mean}")
    print()
    # plt.show()


if __name__ == "__main__":
    training()
