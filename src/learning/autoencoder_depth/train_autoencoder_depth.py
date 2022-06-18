import argparse
import datetime
import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import wandb

import src.learning.residual.architectures as arch
from src.learning.autoencoder_depth.Models.model_CoPhy_like import Autoencoder
from src.learning.autoencoder_depth.Models.model_IRS_like import NormNetS

import src.learning.autoencoder_depth.dataloaders.irs_dataset as irs
import src.learning.autoencoder_depth.dataloaders.euroc_mav_loader as euroc
import src.utils.path as path


class EncoderDecoder(nn.Module):
    def __init__(self,
                 activation: str,
                 embedding_size: int,
                 encoding: str,
                 encoder_channel: int,
                 encoder_layer: int,
                 l2: float,
                 lr: float,
                 model: str,
                 ):
        """
        Model to predict residual z such that f_aero(z, motor). Sequences of depth-maps from past time-steps.

        Args:
            activation: The activation function to use.
            encoding: The encoding method to use. Can be depth or normal.
        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu
            raise NotImplemented
        assert encoding in ['depth', 'normal']
        self.encoding = encoding
        self.encoder_channel = encoder_channel
        self.encoder_layer = encoder_layer
        self.embedding_size = embedding_size
        self.lr = lr
        self.l2 = l2

        models = {'CoPhy_Like': Autoencoder, 'IRS_Like': NormNetS, 'cnn': EncoderDecoder}
        assert model in models
        self.model = model

        self.depth_size = (480, 640)
        from src.learning.residual.architectures import CNNEncoder
        self.encoder = CNNEncoder(embedding_size, layer_num=self.encoder_layer, channel_num=self.encoder_channel)
        if self.encoding == 'depth':
            self.decoder = arch.CNNDecoder(embedding_size, layer_num=self.encoder_layer,
                                           channel_num=self.encoder_channel, out_dim=(480, 640, 1))
        else:
            self.decoder = arch.CNNDecoder(embedding_size, layer_num=self.encoder_layer,
                                           channel_num=self.encoder_channel, out_dim=(480, 640, 3))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute([0, 3, 1, 2])
        x = self.encoder(x)  # (batch_size, embedding_size)
        x = self.decoder(x)  # (batch_size, channel, 480, 640)
        x = x.permute([0, 2, 3, 1])
        return x


def compute_grad(rgb, device):
    rgb = rgb.permute([0, 3, 1, 2])
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).view(1, 1, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).view(1, 1, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 + F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)
    return sobel


def loss_depth(gt, hat):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    mse = ((gt - hat) ** 2).mean()
    mask = gt != 0
    mse_normed = (((gt - hat)[mask] / gt[mask]) ** 2).mean()
    mse_grad = ((compute_grad(gt, device) - compute_grad(hat, device)) ** 2).mean()
    psnr = 10 * torch.log10(24 ** 2 / mse)
    return {
        # 'loss_depth': mse + mse_grad / 2000,
        'loss_depth': mse_normed,
        'mse_depth': mse,
        'mse_grad_depth': mse_grad,
        'mse_normed': mse_normed,
        'psnr_depth': psnr,
    }


def loss_normal(gt, prediction):
    device = 'gpu' if torch.cuda.is_available() else "cpu"
    mse_normal = ((gt - prediction["output"]) ** 2).mean()
    mse_grad = ((compute_grad(gt, device) - compute_grad(prediction["output"], device)) ** 2).mean()
    gt_transpose = gt.permute(0, 2, 3, 1).reshape(-1, 3)
    prediction_transpose = prediction["output"].permute(0, 2, 3, 1).reshape(-1, 3)
    cross_product_loss = torch.sqrt((torch.cross(gt_transpose, prediction_transpose, dim=-1) ** 2).sum(-1)).mean()
    psnr = 10 * torch.log10(4 / mse_normal)
    return {'loss_normal': mse_normal + mse_grad / 2000, 'mse_normal': mse_normal,
            'mse_grad_normal': mse_grad, 'cross_product_normal': cross_product_loss, 'psnr_normal': psnr}


SAVING_PATH = f'{path.ANALYSIS_PATH}irs_encoder_decoder/'
path.mkdir([SAVING_PATH])


def train_valid_irs_auto_encoder():
    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Train EncoderDecoder.')
    parser.add_argument('--activation', default="relu", type=str, help="The activation function to use.")
    parser.add_argument('--batch_size', default=2, type=int, help='Size of a batch.')
    parser.add_argument('--dataset', default="irs", type=str)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--encoder_channel', default=152, type=int, help='Number of channels in the encoder.')
    parser.add_argument('--encoder_layer', default=6, type=int, help='Number of layers in the encoder.')
    parser.add_argument('--encoding', default='depth', type=str, help='"depth" or "normal" supervision for the decoder')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epoch to do.')
    parser.add_argument('--l2', default=0, type=float, help="Number of past information to take in the model.")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate.")
    parser.add_argument('--model', default="cnn", type=str, help='CoPhy_Like, IRS_Like or cnn')
    parser.add_argument('--resume', default='20220615_120420', type=str, help='If training resumes from a checkpoint')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = {'eurocmav': euroc.EurocMavLoader, 'irs': irs.IRSDataset}

    # Logging initialization
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.resume != 'None' and args.resume != 'none':
        # now = args.resume  # TODO
        resume = True
    else:
        resume = False
    version = f'{now}'
    name = f'{EncoderDecoder.__name__}'
    training_path = f'{SAVING_PATH}{name}/{version}/'
    path.mkdir([training_path])
    identifier = f'{name}_{version}'

    wandb.init(name=identifier,
               id=identifier,
               project='turbulence_residual',
               dir=training_path,
               mode='online' if device == torch.device('cuda') else 'disabled',
               # resume=resume,  # TODO
               )

    print(f'Device used by torch: {device}', flush=True)

    h_prms = {'time_start': now,  # hyperparameters
              'name': name,
              'activation': args.activation,
              'batch_size': args.batch_size,
              'dataset': args.dataset,
              'embedding_size': args.embedding_size,
              'encoder_channel': args.encoder_channel,
              'encoder_layer': args.encoder_layer,
              'encoding': args.encoding,
              'epoch': args.epoch,
              'l2': args.l2,
              'lr': args.lr,
              'model': args.model,
              }
    # Reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # torch.backends.cudnn.deterministic = False
    # torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    h_prms['seed'] = seed

    run = EncoderDecoder(
        activation=h_prms['activation'],
        embedding_size=h_prms['embedding_size'],
        encoder_channel=h_prms['encoder_channel'],
        encoder_layer=h_prms['encoder_layer'],
        encoding=h_prms['encoding'],
        l2=h_prms['l2'],
        lr=h_prms['lr'],
        model=h_prms['model'],
    ).to(device)

    parameters = run.parameters()
    count = 0
    for param in parameters:
        count += param.numel()
    print(f'Total number of parameters {count}', flush=True)
    h_prms['total_parameters'] = count

    # Analysis parameters
    pickle.dump(h_prms, open(f'{training_path}h_prms.pkl', 'wb'))
    wandb.config.update(h_prms)
    print(f'Hyperparameters: {h_prms}', flush=True)
    print(f'training_path={training_path}', flush=True)
    print(f'name={name} | version={version}', flush=True)
    # wandb.watch(run)

    num_workers = 15 if device == 'cuda' else 0
    # Datasets

    list_to_load = ['depth']
    if h_prms['encoding'] == 'normal':
        list_to_load.append('normal')
    train_data = DataLoader(irs.IRSDataset('train', list_to_load),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True,
                            worker_init_fn=seed_worker)

    validation_data = DataLoader(irs.IRSDataset('test', list_to_load),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True,
                                 worker_init_fn=seed_worker)

    # Train loop
    memory = np.inf
    log_interval = 50
    running_loss = {}
    optimizer = run.configure_optimizers()
    epoch = 0
    if resume:
        print('Resuming from checkpoint', flush=True)
        model_weights = torch.load(f'{training_path}model.nn', map_location=device)
        run.load_state_dict(model_weights)
        epoch = 12  # TODO
        # checkpoint = torch.load(f'{training_path}checkpoint.pt', map_location=device)
        # model_state_dict = checkpoint['model']
        # run.load_state_dict(model_state_dict)
        # epoch = checkpoint['epoch'] + 1
        # memory = checkpoint['memory']
        # optimizer.load_state_dict(checkpoint['optimizer'])
    for epoch_idx in range(epoch, args.epoch):
        run.train()
        for batch_idx, batch in enumerate(tqdm(train_data, desc=f'Epoch {epoch_idx}')):
            depth = batch['depth'].to(device)
            if h_prms['encoding'] == 'depth':
                hat = run(depth)
                losses = loss_depth(depth, hat)
                loss = losses['loss_depth']
            else:
                normal = batch['normal'].to(device)
                hat = run(depth)
                losses = loss_normal(normal, hat)
                loss = losses['loss_normal']

            for key in losses:
                if batch_idx == 0:
                    running_loss[f'train_{key}'] = []
                running_loss[f'train_{key}'].append(losses[key].detach().cpu().numpy().reshape(1))

            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                for key in losses:
                    wandb.log({f'train_{key}': np.mean(np.concatenate(running_loss[f'train_{key}']))},
                              step=epoch_idx * len(train_data) + batch_idx)
                    running_loss[key] = []
        for key in running_loss:
            running_loss[key] = []
        # Free memory
        del batch, depth, losses, hat, loss
        if h_prms['encoding'] == 'normal':
            del normal
        torch.cuda.empty_cache()

        # validate the model
        run.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_data):
                depth = batch['depth'].to(device)
                if h_prms['encoding'] == 'depth':
                    hat = run(depth)
                    losses = loss_depth(depth, hat)
                    val_loss_key = 'loss_depth'
                else:
                    normal = batch['normal'].to(device)
                    hat = run(depth)
                    losses = loss_normal(normal, hat)
                    val_loss_key = 'loss_normal'
                for key in losses:
                    if batch_idx == 0:
                        running_loss[f'val_{key}'] = []
                    running_loss[f'val_{key}'].append(losses[key].detach().cpu().numpy().reshape(1))
            val_loss = np.mean(np.concatenate(running_loss[f'val_{val_loss_key}']))
            for key in losses:
                wandb.log({f'val_{key}': np.mean(np.concatenate(running_loss[f'val_{key}']))},
                          step=(epoch_idx + 1) * len(train_data))
                running_loss[key] = []
            # Saving the weights
            if val_loss < memory:
                memory = val_loss
                torch.save({
                    'model': run.state_dict(),
                    'epoch': epoch_idx,  # we have finished this epoch_idx
                    'memory': memory,
                    'optimizer': optimizer.state_dict(),
                }, f'{training_path}checkpoint.pt')
                wandb.save(f'{training_path}checkpoint.pt')
        # Free memory
        del batch, depth, losses, hat, val_loss
        if h_prms['encoding'] == 'normal':
            del normal
        torch.cuda.empty_cache()
        wandb.log({'epoch': epoch_idx})


def main():
    train_valid_irs_auto_encoder()


if __name__ == "__main__":
    main()
