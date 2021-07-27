import torch
import argparse
from Dataloaders.EurocMAVLoader import EurocMavLoader
from Dataloaders.IRSLoader import IRSLoader
import torch.nn as nn
from Models.model_CoPhy_like import Autoencoder
from Models.model_IRS_like import NormNetS
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--dataset', default="irs", type=str)
parser.add_argument('--model', default="CoPhy_Like", type=str)
parser.add_argument('--feature_size', default=32, type=int)
parser.add_argument('--weight_grad', default=1, type=int)
parser.add_argument('--name', default='autoencoder_depth_irs_32_irs', type=str)

args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()

ZMAX = 7.48

datasets = {'eurocmav': EurocMavLoader, 'irs': IRSLoader}
models = {'CoPhy_Like': Autoencoder, 'IRS_Like': NormNetS}


def evaluate():
    print(args)
    dataloader = DataLoader(datasets[args.dataset](split='test'), batch_size=16,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    model = models[args.model](feature_size=args.feature_size, in_channel=1, out_channel=1).to(device)
    try:
        state_dict = torch.load(
            f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/{args.model}/{args.name}.nn",
            map_location=device)
    except FileNotFoundError:
        state_dict = torch.load(
            f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/{args.model}/{args.name}.nn",
            map_location=device)

    model.load_state_dict(state_dict)

    validate(model, dataloader, viz=False)


def vizualize(gt, pred):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    for k in range(3):
        ax[k].axis("off")

    gt = gt[0, 0].numpy()
    pred = pred["output"][0, 0].numpy()

    gt = ((gt + 1) / 2) * ZMAX
    pred = ((pred + 1) / 2) * ZMAX

    ax[0].imshow(gt)
    ax[1].imshow(pred)
    ax[2].imshow((pred - gt) ** 2)

    ax[0].set_title("Ground Truth")
    ax[1].set_title("Reconstruction")
    ax[2].set_title("MSE")

    fig.tight_layout()
    plt.show()


def validate(model, dataloader, viz=False, epoch=1000):
    model.eval()
    l_cost, l_depth, l_grad, l_psnr = [], [], [], []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            depthmap = x['depth'].to(device)
            out = model(depthmap)
            costs = get_loss(depthmap, out)
            psnr = 10 * torch.log10(4 / costs[1])

            l_cost.append(costs[0].detach().cpu().numpy())
            l_depth.append(costs[1].detach().cpu().numpy())
            l_grad.append(costs[2].detach().cpu().numpy())
            l_psnr.append(psnr.detach().cpu().numpy())

            if viz == True:
                vizualize(depthmap, out)
    print(
        f"!*  loss : {np.mean(l_cost):.4f},  MSE : {np.mean(l_depth):.4f}, Grad : {np.mean(l_grad):.4f}, PSNR : {np.mean(l_psnr):.2f} dB")
    loss = np.mean(l_cost)
    return loss


def compute_grad(rgb):
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).view(1, 1, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).view(1, 1, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 + F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)
    return sobel


def get_loss(gt, prediction):
    pred = prediction["output"][:, :, :gt.shape[-2], :gt.shape[-1]]
    mse_normal = MSE(gt, pred)
    mse_grad = MSE(compute_grad(gt), compute_grad(pred))
    return 1000 * mse_normal + args.weight_grad * mse_grad, mse_normal, mse_grad


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    train_dataloader = DataLoader(datasets[args.dataset](split='train'), batch_size=16,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)

    val_dataloader = DataLoader(datasets[args.dataset](split='val'), batch_size=16,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)

    model = models[args.model](feature_size=args.feature_size, in_channel=1, out_channel=1).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    memory = 100000

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            depthmap = x['depth'].to(device)

            out = model(depthmap)
            costs = get_loss(depthmap, out)

            optim.zero_grad()
            costs[0].backward()
            optim.step()

        error = validate(model, val_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            try:
                torch.save(model.state_dict(),
                           f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/{args.model}/{args.name}.nn")
            except FileNotFoundError:
                torch.save(model.state_dict(),
                           f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/{args.model}/{args.name}.nn")
            print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
