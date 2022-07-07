import numpy as np
from tqdm import tqdm

import pickle
import sys
import torch
import torchvision.transforms.functional as f
from typing import Tuple

from src.learning.residual.height_estimator import HeightEstimator
import src.data.habitat.loading as habitat
import src.data.euroc_mav.loading2 as euroc_mav
import src.utils.path as path


def get_trained_model_and_dataset(prefix: str, old: bool, dataset_name: str):
    """

    Args:
        prefix: ['none_raw',
         'none_gru',
         'depth_gru',
         'depth_gru_finetune',
         'norm_gru',
         'norm_gru_finetune']
        old: True will load submission models, False will load new models

        dataset_name: 'euroc_mav' or 'habitat'

    Returns: Loaded model, hyperparameter dict and model's name.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if prefix == 'none_raw':
        if old:
            from src.learning.residual.all_height_no_gru import AllHeightNoGRU as Model
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/AllHeight/20220224_095722/'
            model_name = 'epoch=28-step=18936-val_loss=0.57.ckpt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = Model.load_from_checkpoint(f'{wd_path}{model_name}',
                                               activation=h_prms['activation'],
                                               l2=h_prms['l2'],
                                               learning_rate=h_prms['lr'],
                                               uav_model=None,
                                               motor_size=h_prms['motor_size'],
                                               past=h_prms['past'])
        else:
            from src.learning.residual.old.height_estimator import HeightEstimator
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_cnn/20220529_210051/'
            model_name = 'model.nn'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            # h_prms = pickle.dump(open(f'{wd_path}h_prms.pkl', 'wb'))
            model = HeightEstimator(
                activation=h_prms['activation'],
                dropout=h_prms['dropout'],
                embedding_method=h_prms['embedding_method'],
                l2=h_prms['l2'],
                lr=h_prms['lr'],
                motor_size=h_prms['motor_size'],
                past=h_prms['past'],
                                    )
            model.load_state_dict(torch.load(f'{wd_path}{model_name}', map_location=device))
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, False)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, False)

    elif prefix == 'none_gru':
        if old:
            from src.learning.residual.height_estimator import HeightEstimator as Model
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/AllHeight/20220223_170234/'
            model_name = 'epoch=21-step=14365-val_loss=0.51.ckpt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = Model.load_from_checkpoint(f'{wd_path}{model_name}',
                                               activation=h_prms['activation'],
                                               l2=h_prms['l2'],
                                               learning_rate=h_prms['lr'],
                                               uav_model=None,
                                               motor_size=h_prms['motor_size'],
                                               past=h_prms['past'])
        else:
            from src.learning.residual.old.height_estimator import HeightEstimator
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_cnn/20220530_120616/'
            model_name = 'model.nn'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = HeightEstimator(activation=h_prms['activation'],
                                    dropout=h_prms['dropout'],
                                    embedding_method=h_prms['embedding_method'],
                                    l2=h_prms['l2'],
                                    lr=h_prms['lr'],
                                    motor_size=h_prms['motor_size'],
                                    past=h_prms['past'])
            model.load_state_dict(torch.load(f'{wd_path}{model_name}', map_location=device))
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, False)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, False)

    elif prefix == 'depth_gru':
        if old:
            from src.learning.residual.emb_height import EmbHeight as Model
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220223_121251/'
            model_name = 'epoch=136-step=13425-val_loss=0.50.ckpt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = Model.load_from_checkpoint(f'{wd_path}{model_name}',
                                               activation=h_prms['activation'],
                                               embedding=h_prms['embedding'],
                                               l2=h_prms['l2'],
                                               learning_rate=h_prms['lr'],
                                               uav_model=None,
                                               motor_size=h_prms['motor_size'],
                                               past=h_prms['past'])
        else:
            from src.learning.residual.old.height_estimator import HeightEstimator
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_irs_depth_head/20220529_210139/'
            model_name = 'model.nn'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = HeightEstimator(activation=h_prms['activation'],
                                    dropout=h_prms['dropout'],
                                    embedding_method=h_prms['embedding_method'],
                                    l2=h_prms['l2'],
                                    lr=h_prms['lr'],
                                    motor_size=h_prms['motor_size'],
                                    past=h_prms['past'])
            model.load_state_dict(torch.load(f'{wd_path}{model_name}', map_location=device))
        if dataset_name == 'euroc_mav':
            print(f'name: {prefix} | dataset_name: {dataset_name} | Emb not generated')
            return
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], False, True, False, False)

    elif prefix == 'norm_gru':
        if old:
            from src.learning.residual.emb_height2 import EmbHeight as Model
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220227_200520/'
            model_name = 'epoch=90-step=8917-val_loss=0.54.ckpt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = Model.load_from_checkpoint(f'{wd_path}{model_name}',
                                               activation=h_prms['activation'],
                                               embedding=h_prms['embedding'],
                                               l2=h_prms['l2'],
                                               learning_rate=h_prms['lr'],
                                               uav_model=None,
                                               motor_size=h_prms['motor_size'],
                                               past=h_prms['past'])
        else:
            from src.learning.residual.old.height_estimator import HeightEstimator
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_irs_normal_head/20220529_210237/'
            model_name = 'model.nn'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = HeightEstimator(activation=h_prms['activation'],
                                    dropout=h_prms['dropout'],
                                    embedding_method=h_prms['embedding_method'],
                                    l2=h_prms['l2'],
                                    lr=h_prms['lr'],
                                    motor_size=h_prms['motor_size'],
                                    past=h_prms['past'])
            model.load_state_dict(torch.load(f'{wd_path}{model_name}', map_location=device))
        if dataset_name == 'euroc_mav':
            print(f'name: {prefix} | dataset_name: {dataset_name} | Emb not generated')
            return
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], False, False, True, False)

    elif prefix == 'depth_gru_finetune':
        if old:
            # from src.drone_model.residual.fine_tuning2 import FineTunning as Model
            # path = f'{path.WD_PATH}analysis/habitat_residual_training/FineTunning/20220224_183259/'
            # model_name = 'epoch=0-step=49005-val_loss=0.49.ckpt'
            from src.learning.residual.fine_tuning import FineTunning as Model
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/FineTunning/20220226_183121/'
            model_name = 'epoch=2-step=147017-val_loss=0.48.ckpt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = Model.load_from_checkpoint(f'{wd_path}{model_name}',
                                               activation=h_prms['activation'],
                                               l2=h_prms['l2'],
                                               learning_rate=h_prms['lr'],
                                               uav_model=None,
                                               motor_size=h_prms['motor_size'],
                                               past=h_prms['past'])
        else:
            from src.learning.residual.height_estimator import HeightEstimator
            # wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_irs_depth_finetune/20220530_134811/'
            # model_name = 'model.nn'
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_irs_depth_finetune/20220620_154023/'
            model_name = 'checkpoint.pt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = HeightEstimator(
                activation=h_prms['activation'],
                dropout=h_prms['dropout'],
                embedding_method=h_prms['embedding_method'],
                l2=h_prms['l2'],
                regression_size=h_prms['regression_size'],
                cnn_layer_num=h_prms['cnn_layer_num'],
                cnn_channel_num=h_prms['cnn_channel_num'],
                lr=h_prms['lr'],
                motor_size=h_prms['motor_size'],
                past=h_prms['past'],
            )
            model.load_state_dict(torch.load(f'{wd_path}{model_name}', map_location=device)['model'])
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, False)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, False)

    else:
        assert prefix == 'norm_gru_finetune'
        if old:
            # from src.drone_model.residual.fine_tuning2 import FineTunning as Model
            # path = f'{path.WD_PATH}analysis/habitat_residual_training/FineTunning/20220224_183259/'
            # model_name = 'epoch=0-step=49005-val_loss=0.49.ckpt'
            from src.learning.residual.fine_tuning2 import FineTunning as Model
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/FineTunning/20220228_092516/'
            model_name = 'epoch=0-step=6124-val_loss=0.51.ckpt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = Model.load_from_checkpoint(f'{wd_path}{model_name}',
                                               activation=h_prms['activation'],
                                               l2=h_prms['l2'],
                                               learning_rate=h_prms['lr'],
                                               uav_model=None,
                                               motor_size=h_prms['motor_size'],
                                               past=h_prms['past'])
        else:
            from src.learning.residual.height_estimator import HeightEstimator
            wd_path = f'{path.WD_PATH}analysis/habitat_residual_training/HeightEstimator_irs_normal_finetune/20220629_222109/'
            model_name = 'checkpoint.pt'
            h_prms = pickle.load(open(f'{wd_path}h_prms.pkl', 'rb'))
            model = HeightEstimator(
                activation=h_prms['activation'],
                dropout=h_prms['dropout'],
                embedding_method=h_prms['embedding_method'],
                l2=h_prms['l2'],
                regression_size=h_prms['regression_size'],
                cnn_layer_num=h_prms['cnn_layer_num'],
                cnn_channel_num=h_prms['cnn_channel_num'],
                lr=h_prms['lr'],
                motor_size=h_prms['motor_size'],
                past=h_prms['past'],
            )
            model.load_state_dict(torch.load(f'{wd_path}{model_name}', map_location=device)['model'])
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, False)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, False)

    suffix = 'old' if old else 'new'
    name = f'{prefix}_{suffix}'
    return model, h_prms, name, dataset


def model_predict(model, h_prms, name):
    # Data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    # False if '/Users/quentin' in path.WD_PATH else True
    index = habitat.build_index_valid_flight(True)

    # Model
    model.to(device)
    model.eval()

    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std
    t_z_mean = torch.tensor(z_mean).to(device)
    t_z_std = torch.tensor(z_std).to(device)

    with torch.no_grad():
        for scene_number, traj in tqdm(index.items()):
            for flight_number in traj:
                if ('none_raw' in name or 'none_gru' in name or 'depth_gru_finetune' in name
                        or 'norm_gru_finetune' in name):
                    depth = torch.from_numpy(habitat.load_file('depth', scene_number, flight_number)).float().to(device)
                elif 'depth_gru' in name:
                    depth = torch.from_numpy(habitat.load_file('depth_emb', scene_number,
                                                               flight_number)).float().to(device)
                else:
                    assert 'norm_gru' in name
                    depth = torch.from_numpy(habitat.load_file('norm_emb', scene_number,
                                                               flight_number)).float().to(device)
                z_gt = (habitat.load_file('surface_dist', scene_number, flight_number) - z_mean) / z_std
                z_gt = torch.from_numpy(z_gt).float().to(device)
                label_traj = torch.from_numpy(labels[scene_number][flight_number]).float().to(device)
                length = len(depth)

                label_data = torch.empty(length, h_prms['past']).to(device)
                loss = (torch.empty(length) * np.nan).to(device)
                loss_meter = (torch.empty(length) * np.nan).to(device)
                z_hat = (torch.empty(length, 1) * np.nan).to(device)

                for i in range(h_prms['past'], length + 1):
                    label_data[i - 1] = label_traj[i - h_prms['past']:i]
                    z_hat[i - 1] = model(depth[i - h_prms['past']:i].unsqueeze(0))
                weighted_loss, loss0, loss2, loss3 = model.loss(z_gt[h_prms['past'] - 1:],
                                                                z_hat[h_prms['past'] - 1:],
                                                                label_data[h_prms['past'] - 1:])
                loss[h_prms['past'] - 1:] = weighted_loss
                weighted_loss, loss0, loss2, loss3 = model.loss(z_gt[h_prms['past'] - 1:] * t_z_std + t_z_mean,
                                                                z_hat[h_prms['past'] - 1:] * t_z_std + t_z_mean,
                                                                label_data[h_prms['past'] - 1:])
                loss_meter[h_prms['past'] - 1:] = weighted_loss
                flight_folder = f'{habitat.get_flight_folder(scene_number, flight_number)}pred/'
                path.mkdir([flight_folder])
                np.savez(f'{flight_folder}{name}.npz',
                         loss=loss.cpu().numpy(),
                         loss_meter=loss_meter.cpu().numpy(),
                         z_gt=z_gt.cpu().numpy() * z_std + z_mean,
                         z_hat=z_hat.cpu().numpy() * z_std + z_mean)


def predict_on_all_dataset():
    # for stage in ['none_raw', 'none_gru', 'depth_gru', 'depth_gru_finetune', 'norm_gru', 'norm_gru_finetune']:
    stage = 'norm_gru_finetune'
    model, h_prms, name, dataset = get_trained_model_and_dataset(stage, False, 'habitat')
    model_predict(model, h_prms, name)


def test_new_model_habitat():
    print(f'test_new_model_habitat', flush=True)
    # for stage in ['none_raw', 'none_gru', 'depth_gru', 'depth_gru_finetune', 'norm_gru', 'norm_gru_finetune']:
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    for stage in ['none_raw', 'none_gru', 'depth_gru', 'depth_gru_finetune', 'norm_gru', 'norm_gru_finetune']:
        model, h_prms, name, dataset = get_trained_model_and_dataset(stage, False, 'habitat')

        old_loss_agg = []
        new_loss_agg = []
        loss0_agg = []
        loss2_agg = []
        loss3_agg = []

        for scene_number in tqdm(habitat.TEST, desc=f'{stage}', file=sys.stdout):
            for flight_number in labels[scene_number]:
                flight_folder = f'{habitat.get_flight_folder(scene_number, flight_number)}pred/'
                file = np.load(f'{flight_folder}{name}.npz')
                z_gt = file['z_gt']
                z_hat = file['z_hat']

                label = np.stack([labels[scene_number][flight_number][i - h_prms['past']:i] for i in
                                  range(h_prms['past'], len(z_gt) + 1)])
                data_usable = np.logical_not(
                    np.any(label == 1, axis=1))  # check for valid data, label 1 is invalid data
                where_is_label0 = np.logical_and(data_usable, (label == 0)[:, -1])
                where_is_label2 = np.logical_and(data_usable, (label == 2)[:, -1])
                where_is_label3 = np.logical_and(data_usable, (label == 3)[:, -1])

                old_loss = (z_gt[h_prms['past'] - 1:] - z_hat[h_prms['past'] - 1:]) ** 2
                old_loss_agg.append(old_loss)
                new_loss_agg.append(old_loss[data_usable])
                loss0_agg.append(old_loss[where_is_label0])
                loss2_agg.append(old_loss[where_is_label2])
                loss3_agg.append(old_loss[where_is_label3])

        print(f'{name}: old loss: {np.mean(np.concatenate(old_loss_agg)) * 10000:.0f}')
        print(f'{name}: new loss: {np.mean(np.concatenate(new_loss_agg)) * 10000:.0f}')
        print(f'{name}: loss0: {np.mean(np.concatenate(loss0_agg)) * 10000:.0f}')
        print(f'{name}: loss2: {np.mean(np.concatenate(loss2_agg)) * 10000:.0f}')
        print(f'{name}: loss3: {np.mean(np.concatenate(loss3_agg)) * 10000:.0f}')
        print(f'\n\n')


def main():
    # predict_on_all_dataset()
    test_new_model_habitat()


if __name__ == '__main__':
    main()
