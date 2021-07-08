import torch
import argparse
from src.autoencoder_depth.dataloader import Depthmap_Loader
import torch.nn as nn
from src.autoencoder_depth.model import Autoencoder
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--feature_size', default=1024, type=int)
parser.add_argument('--name', default='autoencoder_depth', type=str)

args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()


def evaluate():
    pass


def validate(model, dataloader, viz=False, epoch=1000):
    model.eval()
    l_cost, l_psnr = [], []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            depthmap = x.to(device).unsqueeze(1)

            out = model(depthmap)
            cost = get_loss(depthmap, out)
            psnr = 10 * torch.log10(4 / cost)

            l_cost.append(cost.detach().cpu().numpy())
            l_psnr.append(psnr.detach().cpu().numpy())
    print(f"!*  loss : {np.mean(l_cost):.4f},  PSNR : {np.mean(l_psnr):.2f} dB")
    loss = np.mean(l_cost)
    return loss


def get_loss(gt, predicition):
    return MSE(gt, predicition["output"])


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    train_dataloader = DataLoader(Depthmap_Loader(split='train'), batch_size=2,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)

    ######################
    val_dataloader = DataLoader(Depthmap_Loader(split='train'), batch_size=2,  #############@
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)

    model = Autoencoder(feature_size=args.feature_size).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    memory = 100000

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            depthmap = x.to(device).unsqueeze(1)

            out = model(depthmap)
            cost = get_loss(depthmap, out)

            optim.zero_grad()
            cost.backward()
            optim.step()

        error = validate(model, val_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            torch.save(model.state_dict(), "autoencoder.nn")
            print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
