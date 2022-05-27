import numpy as np
import pickle
import torch
from tqdm import tqdm

import src.data.habitat.loading as habitat
import src.utils.path as path
from src.drone_models.residual.height_estimator import get_trained_model


def model_predict():
    # Data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    # False if '/Users/quentin' in path.WD_PATH else True
    index = habitat.build_index_valid_flight(True)

    # Model
    model, h_prms, name = get_trained_model('none_raw')
    model.to(device)
    model.eval()

    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std
    t_z_mean = torch.tensor(z_mean).to(device)
    t_z_std = torch.tensor(z_std).to(device)

    with torch.no_grad():
        for scene_number, traj in tqdm(index.items()):
            for flight_number in traj:
                depth = torch.from_numpy(habitat.load_file('depth', scene_number, flight_number)).float().to(device)
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
                loss[h_prms['past'] - 1:] = model.loss(z_gt[h_prms['past'] - 1:], z_hat[h_prms['past'] - 1:],
                                                       label_data[h_prms['past'] - 1:])
                loss_meter[h_prms['past'] - 1:] = model.loss(z_gt[h_prms['past'] - 1:] * t_z_std + t_z_mean,
                                                             z_hat[h_prms['past'] - 1:] * t_z_std + t_z_mean,
                                                             label_data[h_prms['past'] - 1:])
                flight_folder = f'{habitat.get_flight_folder(scene_number, flight_number)}pred/'
                path.mkdir([flight_folder])
                np.savez(f'{flight_folder}{name}.npz',
                         loss=loss.cpu().numpy(),
                         loss_meter=loss_meter.cpu().numpy(),
                         z_gt=z_gt.cpu().numpy() * z_std + z_mean,
                         z_hat=z_hat.cpu().numpy() * z_std + z_mean)


def _debug_remove_wrong_predict():
    import os
    model_part_path = 'AllHeight/20220513_154038/'
    model_ckpt = 'epoch=78-step=894042-val_loss=0.04.ckpt'
    name = f'''{model_part_path.replace('/', '_')}{model_ckpt[:-5]}'''
    index = habitat.build_index_valid_flight(True)
    for scene_number, traj in tqdm(index.items()):
        for flight_number in traj:
            flight_folder = f'{habitat.get_flight_folder(scene_number, flight_number)}pred/'
            try:
                os.remove(f'{flight_folder}{name}.npz')
            except:
                print(f'scene_number: {scene_number}, flight_number: {flight_number} already removed')


def main():
    model_predict()


if __name__ == '__main__':
    main()
