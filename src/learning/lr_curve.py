import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.drone_models.residual.all_height as model_builder
import src.data.habitat.loading as habitat
import src.utils.path as path


def main():
    past = 1
    train_dataset = habitat.HabitatResidualDataset('train', past, True, False, False, False, True)
    train_dataloader = DataLoader(train_dataset, 20, True)

    power = torch.linspace(-9, 0, 100)
    lr_test = 10 ** power
    sample_size = 20

    model = model_builder.AllHeight('relu', 0.5, 0, lr_test[0], 4, past)
    model.train()
    loss_mean = torch.zeros_like(lr_test)
    loss_std = torch.zeros_like(lr_test)
    for i, lr in enumerate(tqdm(lr_test)):
        tmp_loss = torch.zeros(sample_size)
        model.lr = lr
        optimizer = model.configure_optimizers()
        for j in range(sample_size):
            batch = train_dataloader.__iter__().__next__()
            loss = model.training_step(batch, 1)
            tmp_loss[j] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_mean[i] = tmp_loss.mean()
        loss_std[i] = tmp_loss.std()

    pickle.dump({'loss_mean': loss_mean, 'loss_std': loss_std}, open(f'{path.WD_PATH}lr_curve.pkl', 'wb'))
    # plt.plot(lr_test, loss_mean)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.savefig('/Users/quentin/Desktop/lr_curve.png')
    # plt.show()


if __name__ == '__main__':
    main()
