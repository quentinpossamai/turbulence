import matplotlib.pyplot as plt
import pickle
import torch

import src.data.habitat.loading as habitat
# from src.drone_models.residual.fine_tuning2 import FineTunning
from src.drone_models.residual.emb_height import EmbHeight as Model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = f'{habitat.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220223_121251/'
    model_name = 'epoch=136-step=13425-val_loss=0.50.ckpt'
    h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
    model = Model.load_from_checkpoint(f'{path}{model_name}',
                                       activation=h_prms['activation'],
                                       embedding=h_prms['embedding'],
                                       l2=h_prms['l2'],
                                       learning_rate=h_prms['lr'],
                                       uav_model=None,
                                       motor_size=h_prms['motor_size'],
                                       past=h_prms['past']).to(device)
    model.eval()
    past = h_prms['past']

    index = habitat.build_index_valid_flight(True)
    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std
    for scene_number, traj_dict in index.items():
        for flight_number, length in traj_dict.items():
            depth_emb = torch.from_numpy(habitat.load_file('depth_emb', scene_number, flight_number)).to(device)
            z_gt = torch.from_numpy(habitat.load_file('surface_dist', scene_number, flight_number)).to(device).squeeze()

            z_hat = torch.empty(len(z_gt) - past) * torch.nan
            with torch.no_grad():
                for i in range(past, len(depth_emb)):
                    z_hat[i - past] = model.forward(depth_emb[i - past: i])
                z_hat = z_hat * z_std + z_mean
                loss = (z_gt[past:] - z_hat) ** 2
            assert not torch.isnan(z_hat).any()
            pickle.dump([z_hat, loss], open(f'{habitat.DATA_HABITAT_PATH}scene_datasets/mp3d/'
                                            f'{habitat.MP3D_SCENE_LIST[scene_number]}/flight2/{flight_number}/'
                                            f'emb_prediction1.pkl', 'wb'))


if __name__ == '__main__':
    main()
