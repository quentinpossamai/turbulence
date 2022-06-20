import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.utils.path as path


def create_lr_tests():
    # Change dataset
    import src.learning.autoencoder_depth.dataloaders.irs_dataset as irs_dataset
    train_dataset = irs_dataset.IRSDataset('train', ['depth'])
    train_dataloader = DataLoader(train_dataset, 20, True, pin_memory=True, num_workers=0)

    power = torch.linspace(-9, 0, 50)
    lr_test = 10 ** power
    sample_size = 1000

    # Change model
    import src.learning.autoencoder_depth.train_autoencoder_depth as auto_encoder
    model = auto_encoder.EncoderDecoder(
        'relu',
        128,
        'depth',
        64,
        6,
        0.05,
        lr_test[0],
        'cnn',
        1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    loss_mean = torch.zeros_like(lr_test)
    loss_std = torch.zeros_like(lr_test)
    for i, lr in enumerate(tqdm(lr_test)):
        tmp_loss = torch.zeros(sample_size)
        model.lr = lr
        optimizer = model.configure_optimizers()
        for j in range(sample_size):
            batch = train_dataloader.__iter__().__next__()
            batch = {key: value.to(device) for key, value in batch.items()}
            hat = model(batch['depth'])
            loss = auto_encoder.loss_depth(batch['depth'], hat, 1)['loss_depth']
            tmp_loss[j] = loss.cpu().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_mean[i] = tmp_loss.mean()
        loss_std[i] = tmp_loss.std()

    pickle.dump({'loss_mean': loss_mean, 'loss_std': loss_std}, open(f'{path.WD_PATH}lr_curve_ae.pkl', 'wb'))
    # plt.plot(lr_test, loss_mean)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.savefig('/Users/quentin/Desktop/lr_curve.png')
    # plt.show()


def plot_lr_curve():
    power = torch.linspace(-9, 0, 100)
    lr_test = 10 ** power

    data = pickle.load(open(f'{path.WD_PATH}lr_curve_height_estimator.pkl', 'rb'))
    plt.plot(lr_test, data['loss_mean'])
    plt.plot(lr_test, data['loss_mean'] + data['loss_std'])
    plt.plot(lr_test, data['loss_mean'] - data['loss_std'])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Learning rate')
    plt.ylabel('Training Loss')
    plt.show()


def main():
    # create_lr_tests()
    plot_lr_curve()


if __name__ == '__main__':
    main()
