import argparse
import datetime
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import wandb

from src.learning.autoencoder_depth.Models.model_CoPhy_like import Autoencoder
import src.data.habitat.loading as habitat
import src.utils.path as path


class HeightEstimator(nn.Module):
    def __init__(self, activation: str,
                 dropout: float,
                 embedding_method: str,
                 l2: float,
                 regression_size: int,
                 cnn_layer_num: int,
                 cnn_channel_num: int,
                 lr: float,
                 motor_size: int,
                 past: int,
                 ):
        """
        Model to predict residual z such that f_aero(z, motor). Sequences of depth-maps from past time-steps.

        Args:
            activation: The activation function to use.
            dropout: The dropout probability of MLPs hidden layers.
            l2: Float to penalize all weights with l2 loss.
            lr: Learning rate for Adam optimizer.
            motor_size: The number of motors.
            past: The number of past depth-maps to use.
            embedding_method: The type of pre-training to use.
        """
        super().__init__()
        self.activation = None
        assert activation in ['relu', 'tanh']
        if activation == "relu":
            self.activation = torch.relu  # torch.tanh or f.relu
        elif activation == "tanh":
            self.activation = torch.tanh  # torch.tanh or f.relu
        self.dropout = dropout
        assert embedding_method in ['none', 'cnn', 'irs_depth', 'irs_normal']
        self.embedding_method = embedding_method
        self.l2 = l2
        self.lr = lr
        self.regression_size = regression_size
        self.cnn_layer_num = cnn_layer_num
        self.cnn_channel_num = cnn_channel_num
        self.motor_size = motor_size
        self.past = past

        embedding_size = 128
        import src.learning.autoencoder_depth.train_autoencoder_depth as arch

        if embedding_method == 'none':
            self.embedding_layer = lambda x: x
            self.depth_size = (128,)
        else:
            assert embedding_method in ['cnn', 'irs_depth', 'irs_normal']
            self.depth_size = (1, 480, 640)
            self.embedding_layer = arch.EncoderDecoder(
                activation=activation,
                embedding_size=embedding_size,
                encoding='depth',  # does not matter if it is depth or normal for encoder
                encoder_channel=self.cnn_channel_num,
                encoder_layer=self.cnn_layer_num,
                l2=self.l2,
                lr=self.lr,
                model='cnn',
            ).encoder

        from src.learning.residual.architectures import RegressionHead
        self.regression_head = RegressionHead(self.activation, embedding_size, self.regression_size, self.past,
                                              self.dropout)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        r"""
        Estimate the residual

        param in_data: depth (*, past, width, height)
                       embedding (*, past, 128)
        Generally * means batch dimension or more but must exist during training to compute batch_norm.

        return: (*, 1)
        """
        input_data = input_data.reshape((-1,) + self.depth_size)
        out = self.embedding_layer(input_data)
        out = self.regression_head(out)
        return out

    def loss(self, z_gt, z_hat, label):
        """
        Args:
            z_gt: (batch, 1),
            z_hat: (batch, 1),
            label: (batch, self.past, 1)
            label 0: normal data
            label 1: data to be rejected
            label 2: surface discontinuity data
            label 3: noisy data

        Returns: The loss used by the model.
        """
        data_usable = torch.logical_not(torch.any(label == 1, dim=1))  # check for valid data, label 1 is invalid data
        where_is_label0 = torch.logical_and(data_usable, (label == 0)[:, -1])
        where_is_label2 = torch.logical_and(data_usable, (label == 2)[:, -1])
        where_is_label3 = torch.logical_and(data_usable, (label == 3)[:, -1])

        weight_label2 = 1
        weights_label2_array = data_usable.float() + where_is_label2.float() * (weight_label2 - 1)

        weight_label3 = 1
        weights_label3_array = data_usable.float() + where_is_label3.float() * (weight_label3 - 1)

        initial_loss = ((z_gt - z_hat) ** 2).squeeze()
        weighted_loss = initial_loss * data_usable
        weighted_loss = weighted_loss * weights_label2_array
        weighted_loss = weighted_loss * weights_label3_array

        loss0 = initial_loss[where_is_label0]
        loss2 = initial_loss[where_is_label2]
        loss3 = initial_loss[where_is_label3]

        return weighted_loss, loss0, loss2, loss3


ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING = f'{path.ANALYSIS_PATH}euroc_mav_residual_training/'
ANALYSIS_HABITAT_RESIDUAL_TRAINING = f'{path.ANALYSIS_PATH}habitat_residual_training/'

SAVING_PATH = ANALYSIS_HABITAT_RESIDUAL_TRAINING
path.mkdir([SAVING_PATH])


def train_valid_height_estimator():
    # Args are hyperparameters to optimize
    parser = argparse.ArgumentParser(description='Identify model on data.')
    parser.add_argument('--activation', default="relu", type=str, help="The activation function to use.")
    parser.add_argument('--batch_size', default=2, type=int, help='Size of a batch.')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout probability')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epoch to do.')
    parser.add_argument('--l2', default=0, type=float, help="Number of past information to take in the model.")
    parser.add_argument('--lr', default=5e-5, type=float, help="Learning rate.")
    parser.add_argument('--cnn_layer_num', default=6, type=int, help="Number of layer for the CNN encoder.")
    parser.add_argument('--cnn_channel_num', default=64, type=int, help="Number of channels for the CNN encoder.")
    parser.add_argument('--regression_size', default=50, type=int,
                        help="Number of unit for MLP and hidden_size for GRU.")
    parser.add_argument('--past', default=20, type=int, help="Number of past information to take in the model.")
    parser.add_argument('--stage', default='irs_depth_finetune', type=str, help="The stage of the model.")
    parser.add_argument('--resume', default='none', type=str, help='If training resumes from a checkpoint')

    args = parser.parse_args()

    # ae training is already done and written elsewhere
    assert args.stage in ['cnn',
                          'irs_depth_head', 'irs_depth_finetune',
                          'irs_normal_head', 'irs_normal_finetune']
    embedding_method = args.stage
    if 'irs_depth_head' == args.stage or 'irs_normal_head' == args.stage:
        embedding_method = 'none'  # training only the head. irs embedding is done in autoencoder_depth
    if 'irs_depth_finetune' == args.stage or 'irs_normal_finetune' == args.stage:
        embedding_method = args.stage[:-9]

    # Logging initialization
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.resume != 'None' and args.resume != 'none':
        now = args.resume
        resume = True
    else:
        resume = False
    version = f'{now}'
    residual_model_class = HeightEstimator
    name = f'{residual_model_class.__name__}_{args.stage}'
    training_path = f'{SAVING_PATH}{name}/{version}/'
    path.mkdir([training_path])

    identifier = f'{name}_{version}'
    wandb.init(name=identifier,
               id=identifier,
               project='turbulence_residual',
               dir=training_path,
               mode='online' if sys.platform != 'darwin' else 'disabled')

    use_cuda_if_possible = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')

    print(f'Device used by torch: {device}', flush=True)

    # Display entire dataset
    check_all_file = True  # False if '/Users/quentin' in path.WD_PATH else True
    for split_name in ['train', 'valid', 'test']:
        dataset_stats = habitat.HabitatResidualDataset(split_name, args.past, load_depth=False, load_depth_emb=False,
                                                       load_norm_emb=False, normed_depth=False,
                                                       check_all_file=check_all_file)
        print(split_name)
        for scene_number, scene_traj in dataset_stats.index.items():
            print(f'{scene_number} - {habitat.MP3D_SCENE_LIST[scene_number]}: {scene_traj}')
        print(f'Data length: | {len(dataset_stats)}', flush=True)
        print('\n\n', flush=True)

    # Analysis parameters
    motor_size = 4
    h_prms = {'time_start': now,  # hyperparameters
              'name': name,
              'activation': args.activation,
              'batch_size': args.batch_size,
              'dropout': args.dropout,
              'embedding_method': embedding_method,
              'l2': args.l2,
              'lr': args.lr,
              'cnn_layer_num': args.cnn_layer_num,
              'cnn_channel_num': args.cnn_channel_num,
              'regression_size': args.regression_size,
              'motor_size': motor_size,
              'epoch': args.epoch,
              'past': args.past,
              'stage': args.stage}
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

    run = residual_model_class(activation=h_prms['activation'],
                               dropout=h_prms['dropout'],
                               l2=h_prms['l2'],
                               lr=h_prms['lr'],
                               cnn_layer_num=h_prms['cnn_layer_num'],
                               cnn_channel_num=h_prms['cnn_channel_num'],
                               regression_size=h_prms['regression_size'],
                               motor_size=h_prms['motor_size'],
                               past=h_prms['past'],
                               embedding_method=h_prms['embedding_method']).to(device)
    wandb.watch(run)

    # Load previous weight depending on training stage
    if h_prms['stage'] == 'cnn':
        input_key = 'depth'
        load_depth, load_depth_emb, load_norm_emb, normed_depth = True, False, False, False
    elif h_prms['stage'] in 'irs_depth_head':
        input_key = 'depth_emb'
        load_depth, load_depth_emb, load_norm_emb, normed_depth = False, True, False, False
    elif h_prms['stage'] == 'irs_normal_head':
        input_key = 'norm_emb'
        load_depth, load_depth_emb, load_norm_emb, normed_depth = False, False, True, False
    elif h_prms['stage'] == 'irs_depth_finetune':
        input_key = 'depth'
        import src.learning.autoencoder_depth.predict as ae_models
        if h_prms['cnn_layer_num'] == 6 and h_prms['cnn_channel_num'] == 128 and not resume:
            model, h_prms_ae = ae_models.get_irs_model('20220611_183103')
            run.embedding_layer.load_state_dict(model.encoder.state_dict())
        elif h_prms['cnn_layer_num'] == 6 and h_prms['cnn_channel_num'] == 64 and not resume:
            model, h_prms_ae = ae_models.get_irs_model('20220611_182536')
            run.embedding_layer.load_state_dict(model.encoder.state_dict())
        elif h_prms['cnn_layer_num'] == 9 and h_prms['cnn_channel_num'] == 32 and not resume:
            model, h_prms_ae = ae_models.get_irs_model('20220611_183523')
            run.embedding_layer.load_state_dict(model.encoder.state_dict())
        else:
            raise NotImplementedError

        # Loading head
        # state_dict_path = 'analysis/habitat_residual_training/HeightEstimator_irs_depth_head/20220529_210139/model.nn'
        # state_dict = torch.load(f'{path.WD_PATH}{state_dict_path}', map_location=device)
        # state_dict = {k[16:]: v for k, v in state_dict.items()}
        # run.regression_head.load_state_dict(state_dict)
        load_depth, load_depth_emb, load_norm_emb, normed_depth = True, False, False, False
    else:
        assert h_prms['stage'] == 'irs_normal_finetune'
        import src.learning.autoencoder_depth.predict as ae_models
        input_key = 'depth'
        if h_prms['cnn_layer_num'] == 9 and h_prms['cnn_channel_num'] == 128 and not resume:
            model, h_prms_ae = ae_models.get_irs_model('20220620_173528')
            run.embedding_layer.load_state_dict(model.encoder.state_dict())
        elif h_prms['cnn_layer_num'] == 9 and h_prms['cnn_channel_num'] == 64 and not resume:
            model, h_prms_ae = ae_models.get_irs_model('20220620_173515')
            run.embedding_layer.load_state_dict(model.encoder.state_dict())
            print(f'Reloading previous weights')
        elif h_prms['cnn_layer_num'] == 6 and h_prms['cnn_channel_num'] == 128 and not resume:
            model, h_prms_ae = ae_models.get_irs_model('20220620_173454')
            run.embedding_layer.load_state_dict(model.encoder.state_dict())
        else:
            print(f'No weight reloaded')
        # # Loading head
        # state_dict_path = 'analysis/habitat_residual_training/HeightEstimator_irs_normal_head/20220529_210237/model.nn'
        # state_dict = torch.load(f'{path.WD_PATH}{state_dict_path}', map_location=device)
        # state_dict = {k[16:]: v for k, v in state_dict.items()}
        # run.regression_head.load_state_dict(state_dict)
        load_depth, load_depth_emb, load_norm_emb, normed_depth = True, False, False, False

    parameters = run.parameters()
    count = 0
    for param in parameters:
        count += param.numel()
    print(f'Total number of parameters {count}', flush=True)
    h_prms['total_parameters'] = count

    pickle.dump(h_prms, open(f'{training_path}h_prms.pkl', 'wb'))
    wandb.config.update(h_prms)
    print(f'Hyperparameters: {h_prms}', flush=True)
    print(f'training_path={training_path}', flush=True)
    print(f'name={name} | version={version}', flush=True)

    num_workers = 15 if device == 'gpu' else 0
    # Datasets
    print(f'Load depth {load_depth} | load_depth_emb {load_depth_emb} | '
          f'load_norm_emb {load_norm_emb} | normed_depth {normed_depth}', flush=True)
    train_data = DataLoader(habitat.HabitatResidualDataset('train', past=args.past, load_depth=load_depth,
                                                           load_depth_emb=load_depth_emb,
                                                           load_norm_emb=load_norm_emb,
                                                           normed_depth=normed_depth,
                                                           check_all_file=check_all_file),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True,
                            worker_init_fn=seed_worker)

    validation_data = DataLoader(habitat.HabitatResidualDataset('valid', past=args.past, load_depth=load_depth,
                                                                load_depth_emb=load_depth_emb,
                                                                load_norm_emb=load_norm_emb,
                                                                normed_depth=normed_depth,
                                                                check_all_file=check_all_file),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True,
                                 worker_init_fn=seed_worker)

    # Train loop
    memory = np.inf
    log_interval = 50
    train_log_keys = ['train_loss', 'train_loss0', 'train_loss2', 'train_loss3']
    val_log_keys = ['val_loss', 'val_loss0', 'val_loss2', 'val_loss3']
    running_loss = {}
    for key in train_log_keys + val_log_keys:
        running_loss[key] = []
    optimizer = run.configure_optimizers()
    epoch = 0

    if resume:
        print('Resuming from checkpoint', flush=True)
        checkpoint = torch.load(f'{training_path}checkpoint.pt', map_location=device)
        model_state_dict = checkpoint['model']
        run.load_state_dict(model_state_dict)
        epoch = checkpoint['epoch'] + 1
        memory = checkpoint['memory']
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch_idx in range(epoch, args.epoch):
        run.train()
        for batch_idx, batch in enumerate(tqdm(train_data, desc=f'Epoch {epoch_idx}')):
            model_input = batch[input_key].to(device)
            z_gt = batch['surface_dist'].to(device)
            label = batch['label'].to(device)

            z_hat = run(model_input)
            loss, loss0, loss2, loss3 = run.loss(z_gt, z_hat, label)
            # Logging
            running_loss['train_loss'].append(loss.detach().cpu().numpy())
            running_loss['train_loss0'].append(loss0.detach().cpu().numpy())
            running_loss['train_loss2'].append(loss2.detach().cpu().numpy())
            running_loss['train_loss3'].append(loss3.detach().cpu().numpy())
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                for key in ['train_loss', 'train_loss0', 'train_loss2', 'train_loss3']:
                    wandb.log({key: np.mean(np.concatenate(running_loss[key]))},
                              step=epoch_idx * len(train_data) + batch_idx)
                    running_loss[key] = []
        for key in train_log_keys:
            running_loss[key] = []
        # Free memory
        del batch, model_input, z_gt, label, z_hat, loss, loss0, loss2, loss3
        torch.cuda.empty_cache()

        # validate the model
        run.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_data):
                model_input = batch[input_key].to(device)
                z_gt = batch['surface_dist'].to(device)
                label = batch['label'].to(device)

                z_hat = run(model_input)
                loss, loss0, loss2, loss3 = run.loss(z_gt, z_hat, label)
                running_loss['val_loss'].append(loss.detach().cpu().numpy())
                running_loss['val_loss0'].append(loss0.detach().cpu().numpy())
                running_loss['val_loss2'].append(loss2.detach().cpu().numpy())
                running_loss['val_loss3'].append(loss3.detach().cpu().numpy())
            val_loss = np.mean(np.concatenate(running_loss['val_loss']))
            for key in val_log_keys:
                wandb.log({key: np.mean(np.concatenate(running_loss[key]))},
                          step=(epoch_idx + 1) * len(train_data))
                running_loss[key] = []
            # Saving the weights
            if val_loss < memory:
                memory = val_loss
                torch.save({
                    'model': run.state_dict(),
                    'epoch': epoch_idx,
                    'memory': memory,
                    'optimizer': optimizer.state_dict(),
                }, f'{training_path}checkpoint.pt')
                wandb.save(f'{training_path}checkpoint.pt')

        # Free memory
        del batch, model_input, z_gt, label, z_hat, loss, loss0, loss2, loss3
        torch.cuda.empty_cache()

        wandb.log({'epoch': epoch_idx})


def main():
    train_valid_height_estimator()


if __name__ == "__main__":
    main()
