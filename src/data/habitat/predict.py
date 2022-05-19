import pickle
import torch
from tqdm import tqdm

import src.data.habitat.loading as habitat
import src.utils.path as path


def main():
    # Data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    # False if '/Users/quentin' in path.WD_PATH else True
    index = habitat.build_index_valid_flight(True)

    # Model
    model_path = f'{path.ANALYSIS_PATH}habitat_residual_training/'
    model_part_path = 'AllHeight/20220513_154038/'
    model_ckpt = 'epoch=78-step=894042-val_loss=0.04.ckpt'
    model_path = f'{model_path}{model_part_path}'
    from src.drone_models.residual.all_height import AllHeight
    h_prms = pickle.load(open(f'{model_path}h_prms.pkl', 'rb'))
    model = AllHeight.load_from_checkpoint(f'{model_path}{model_ckpt}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           lr=h_prms['lr'],
                                           uav_model=None,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
    model.eval()
    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std

    with torch.no_grad():
        for scene_number, traj in tqdm(index.items()):
            for flight_number in traj:
                depth = torch.from_numpy(habitat.load_file('depth', scene_number, flight_number)).float().to(device)
                z_gt = (habitat.load_file('surface_dist', scene_number, flight_number) - z_mean) / z_std
                z_gt = torch.from_numpy(z_gt).float().to(device)
                label_traj = torch.from_numpy(labels[scene_number][flight_number]).float().to(device)
                length = len(depth)
                label_data = torch.empty(length, h_prms['past']).to(device)
                loss = (torch.empty(length) * torch.nan).to(device)
                z_hat = (torch.empty(length, 1) * torch.nan).to(device)
                for i in range(h_prms['past'], length):
                    label_data[i - 1] = label_traj[i - h_prms['past']:i]
                    z_hat[i - 1] = model(depth[i - h_prms['past']:i].unsqueeze(0))
                loss[h_prms['past'] - 1:] = model.loss(z_gt[h_prms['past'] - 1:], z_hat[h_prms['past'] - 1:],
                                                       label_data[h_prms['past'] - 1:])
                name = f'''{model_part_path.replace('/', '_')}{model_ckpt[:-5]}'''
                habitat.save_file(f'pred/{name}', {'loss': loss.cpu().numpy(),
                                                   'z_gt': z_gt.cpu().numpy() * z_std + z_mean,
                                                   'z_hat': z_hat.cpu().numpy() * z_std + z_mean},
                                  scene_number, flight_number)


if __name__ == '__main__':
    main()
