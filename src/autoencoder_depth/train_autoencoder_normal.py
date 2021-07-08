import torch
import argparse
from src.autoencoder_depth.dataloaders import Normalmap_Loader
import torch.nn as nn
from src.autoencoder_depth.model import Autoencoder
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--feature_size', default=1024, type=int)
parser.add_argument('--name', default='autoencoder_normal', type=str)

args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()


def evaluate():
    dataloader = DataLoader(Normalmap_Loader(split='train'), batch_size=2,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    model = Autoencoder(feature_size=args.feature_size, in_channel=1, out_channel=3).to(device)
    try:
        state_dict = torch.load(f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/{args.name}.nn",
                                map_location=device)
    except FileNotFoundError:
        state_dict = torch.load(
            f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/{args.name}.nn",
            map_location=device)

    model.load_state_dict(state_dict)

    validate(model, dataloader, viz=True)


def vizualize(gt, pred):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    gt = gt[0].numpy().transpose(1, 2, 0)
    pred = pred["output"][0].numpy().transpose(1, 2, 0)

    gt = (gt + 1) / 2
    pred = (pred + 1) / 2

    for k in range(3):
        ax[k].axis("off")

    ax[0].imshow(gt)
    ax[1].imshow(pred)
    ax[2].imshow(((pred - gt) ** 2).sum(-1))

    ax[0].set_title("Ground Truth")
    ax[1].set_title("Reconstruction")
    ax[2].set_title("MSE")

    fig.tight_layout()
    plt.show()


def validate(model, dataloader, viz=False, epoch=1000):
    model.eval()
    l_cost, l_psnr = [], []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            depthmap = x["depth_estimation"].to(device).unsqueeze(1)
            normal = x['normal'].to(device)

            out = model(depthmap)
            cost = get_loss(normal, out)
            psnr = 10 * torch.log10(4 / cost)

            l_cost.append(cost.detach().cpu().numpy())
            l_psnr.append(psnr.detach().cpu().numpy())

            if viz == True:
                vizualize(depthmap, out)
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

    train_dataloader = DataLoader(Normalmap_Loader(split='train'), batch_size=2,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)

    ######################
    val_dataloader = DataLoader(Normalmap_Loader(split='train'), batch_size=2,  #############@
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

    model = Autoencoder(feature_size=args.feature_size, in_channel=1, out_channel=3).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    memory = 100000

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            depthmap = x['depth_estimation'].to(device).unsqueeze(1)
            normal = x['normal'].to(device)

            out = model(depthmap)
            cost = get_loss(normal, out)

            optim.zero_grad()
            cost.backward()
            optim.step()

        error = validate(model, val_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            try:
                torch.save(model.state_dict(),
                           f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/{args.name}.nn")
            except FileNotFoundError:
                torch.save(model.state_dict(),
                           f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/{args.name}.nn")
            print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
