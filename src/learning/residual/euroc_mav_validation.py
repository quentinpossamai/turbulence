import numpy as np
import torch
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import src.utils.path as path
import src.learning.residual.predict as predict
import src.data.habitat.loading as habitat
import src.data.euroc_mav.loading2 as euroc

z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std
fa_mean, fa_std = habitat.HabitatResidualDataset.fa_mean, habitat.HabitatResidualDataset.fa_std


def euroc_mav_identification():
    dataset = euroc.EurocMAVDatasetHorizon2('train', 1, False, False, False, False)
    from torch.utils.data import DataLoader
    data = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)))

    mask = (np.array(data['flight']) == 'V1_01_easy/')
    motor = data['motor'].numpy()[mask]
    ground_force = (data['ground_force'] * fa_std + fa_mean).numpy()[:, 2:3][mask]
    surface_dist = (data['surface_dist'] * z_std + z_mean).numpy()[mask]

    plt.plot(ground_force)

    import scipy.optimize
    # sol = scipy.optimize.fmin(error, np.ones((2)) * 1e-2)
    def f(x):
        _alpha, _beta = x
        _ground_force_hat = _alpha * np.sum(motor ** 2, axis=1, keepdims=True) / (surface_dist ** 2 - _beta)
        return np.sum((ground_force - _ground_force_hat) ** 2)

    a = scipy.optimize.fmin(f, np.array([1e-2, 1e-2]))
    alpha = a[0]  # 7.410246087055749e-07
    beta = a[1]  # 0.006738302825333955
    ground_force_hat = alpha * np.sum(motor ** 2, axis=1, keepdims=True) / (surface_dist ** 2 - beta)
    plt.plot(ground_force, label='gt')
    plt.plot(ground_force_hat, label='hat')
    plt.legend()
    plt.grid()


def cheeseman_identified(motor, surface_dist):
    # _alpha = 7.410246087055749e-07
    # _beta = 0.006738302825333955

    # alpha = 9.40693193e-07
    # beta = -4.17981623e-03

    alpha = 9.428555120568678e-07
    beta = -0.004331647000391248

    def compute_ground_effect(motor: np.ndarray, z: float) -> np.ndarray:
        sol = [9.40693193e-07, -4.17981623e-03]
        return sol[0] * (motor ** 2).sum(axis=-1) / (z ** 2 - sol[1])

    ground_force_hat = alpha * np.sum(motor ** 2, axis=1, keepdims=True) / (surface_dist ** 2 - beta)
    return ground_force_hat


def euroc_mav_validation(name: str, old: bool):
    # Create torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, h_prms, _, dataset = predict.get_trained_model_and_dataset(name, old, 'euroc_mav')
    model.to(device)

    model.eval()
    pred_dict = {"V1_01_easy/": [], "V1_02_medium/": [], "V1_03_difficult/": []}
    for i in tqdm(range(len(dataset)), desc=f'pred_dict_euroc_mav__{name}_{old}.pkl'):
        # Loading data
        data = dataset[i]
        fa_gt = (data['ground_force'] * fa_std + fa_mean).numpy()[2:3]
        z_gt = (data['surface_dist'] * z_std + z_mean).numpy()
        motor = (data['motor']).numpy()
        flight = data['flight']
        with torch.no_grad():
            # Predicting
            z_hat = model(data['depth'].to(device)).cpu().numpy().flatten() * z_std + z_mean
        fa_ci = cheeseman_identified(motor[np.newaxis, ...], z_gt[np.newaxis, ...]).flatten()
        fa_hat = cheeseman_identified(motor[np.newaxis, ...], z_hat[np.newaxis, ...]).flatten()

        pred_dict[flight].append(np.array([z_gt[0],
                                           z_hat[0],
                                           fa_gt[0],
                                           fa_ci[0],
                                           fa_hat[0],
                                           ]))
    for key in pred_dict.keys():
        pred_dict[key] = np.stack(pred_dict[key], axis=0)
    pickle.dump(pred_dict,
                open(f'{path.WD_PATH}analysis/euroc_mav_validation/pred_dict_euroc_mav__{name}_{old}.pkl', 'wb'))
    print('')


def plot():
    plt.style.use('seaborn-deep')
    pred_dict = pickle.load(
        open(f'{path.WD_PATH}analysis/euroc_mav_validation/pred_dict_euroc_mav__none_gru_old.pkl', 'rb'))

    for flight in pred_dict:
        fa_gt = pred_dict[flight][20:, 2]
        fa_ci = pred_dict[flight][20:, 3]
        fa_hat = pred_dict[flight][20:, 4]
        gt_vs_cheese = np.sqrt(((fa_gt - fa_ci) ** 2).mean())
        gt_vs_ours = np.sqrt(((fa_gt - fa_hat) ** 2).mean())
        print(f'{flight} GT vs Cheeseman: {gt_vs_cheese:.4f}')
        print(f'{flight} GT vs Ours: {gt_vs_ours:.4f}\n\n')

    # Plotting z
    if False:
        plt.figure(figsize=(10, 5))
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, key in enumerate(pred_dict.keys()):
            ax: plt.Axes = axs[i]
            xx = np.arange(pred_dict[key].shape[0]) / 20
            ax.plot(xx, pred_dict[key][:, 0], label='GT')
            ax.plot(xx, pred_dict[key][:, 1], label='Ours')
            ax.set_title(f'{key}')
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Z [m]')
            ax.grid()
        plt.show()

    mask = {'V1_01_easy/': [20, pred_dict['V1_01_easy/'].shape[0]],
            'V1_02_medium/': [20, pred_dict['V1_02_medium/'].shape[0]],
            'V1_03_difficult/': [20, pred_dict['V1_03_difficult/'].shape[0]]}
    # Plotting fa
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, key in enumerate(pred_dict.keys()):
        ax: plt.Axes = axs[i]
        xx = (np.arange(0, mask[key][1] - mask[key][0]) / 20)
        # xx = (np.linspace(0, pred_dict[key].shape[0], mask[key][1] - mask[key][0]) / 20)
        ax.plot(xx, pred_dict[key][mask[key][0]:mask[key][1], 2], label='GT')
        ax.plot(xx, pred_dict[key][mask[key][0]:mask[key][1], 3], label='Cheeseman approx.')
        ax.plot(xx, pred_dict[key][mask[key][0]:mask[key][1], 4], label='Ours')
        ax.set_title(f'{key}')
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force [N]')
        ax.grid()
        ax.set_ylim((-0.7, 2.))
    # plt.savefig(f'{path.WD_PATH}reports/IROS_01_03_2022/graphics/fig6_new_results.png')
    plt.show()


def main():
    # euroc_mav_identification()

    for name in ['none_raw', 'none_gru', 'irs_depth_finetune', 'irs_normal_finetune']:
        for old in [True, False]:
            euroc_mav_validation(name, old)
    name = 'irs_normal_finetune_loss2'
    old = False
    euroc_mav_validation(name, old)
    # plot()


if __name__ == '__main__':
    main()
