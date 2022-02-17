import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import torch
from tqdm import tqdm

import src.data.habitat.loading as habitat
from src.drone_models.residual.gru_cnn_var_d import VariantD
from src.drone_models.residual.gru_cnn_var_a import VariantA
from src.drone_models.analytical_models.quadrotor_4parms import NumpyDynamicalModel
from src.utils.profiler import profiler


def plot_fa_generator():
    cap = 0.1
    mu = 1
    drone_width, drone_height = 0.1, 0.025
    volumetric_mass = 200
    g = 9.81
    m = volumetric_mass * drone_width ** 2 * drone_height
    drone_prms = {'width': drone_width,  # [m]
                  'height': drone_height,  # [m]
                  'm': m,  # [kg]
                  'kt': m * g / (4 * 150 ** 2),  #
                  'km': m * g / (4 * 150 ** 2) * 2 ** 0.5 * drone_width / 2,  #
                  'l': 2 ** 0.5 * drone_width / 2,  # [m]
                  'ix': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iy': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iz': 1 / 3 * (drone_width ** 2 + drone_width ** 2) * m,  # [kg.m2]
                  'prop_radius': drone_width * 0.1}  # [m]
    theta = np.array([drone_prms['kt'] / drone_prms['m'],
                      drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                      drone_prms['km'] / drone_prms['iz'],
                      drone_prms['iz'] / drone_prms['ix']])

    def _apply(_z, _motor):
        ground_force = np.zeros(3)
        if _z < np.sqrt(mu) * drone_prms['prop_radius'] / 4:
            ground_force[2] = cap * drone_prms['m'] * g
        else:
            mu_r_z = mu * (drone_prms['prop_radius'] / (4 * _z)) ** 2
            ground_force[2] = np.clip(np.sum(_motor) ** 2 * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                      a_min=0, a_max=cap * drone_prms['m'] * g)
        return ground_force

    # # Debug plot
    motor = np.array([150, 150, 150, 150])
    z = np.linspace(1e-8, 0.1, 1000)
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})
    for mu in [1, 2, 3, 4, 5]:
        to_plot = np.array([_apply(e, motor) for e in z])[:, 2]
        plt.plot(z, to_plot, label=f'mu={mu}')
    plt.title('Ground effect')
    plt.grid()
    plt.ylabel('Ground effect (N)')
    plt.xlabel('Surface distance (m)')
    plt.legend()
    plt.subplots_adjust(left=0.1, bottom=0.09, right=0.99, top=0.95, wspace=0, hspace=0)
    plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/fa_generator.pdf')
    plt.show()


def plot_fa():
    index = {i: None for i in range(len(habitat.MP3D_SCENE_LIST))}
    for scene_number, scene_name in enumerate(habitat.MP3D_SCENE_LIST):
        index[scene_number] = {}
        try:  # Try finding flight folder
            flights = [int(e) for e in next(os.walk(f'{habitat.MP3D_DATA}{scene_name}/flight/'))[1]]
            for number in flights:
                stats = habitat.load_file('stats', scene_number, number)
                if stats['done'] is not None:  # If the drone has flight at least one step
                    mpc_error = habitat.load_file('mpc_error', scene_number, number)
                    if (mpc_error < 5).all() and len(mpc_error) >= 200:  # If controller is not wrong and traj long
                        # enough
                        index[scene_number][number] = len(mpc_error)
        except StopIteration:
            continue

    fa = []
    z = []
    for scene_number, flight in index.items():
        for flight_number in flight:
            fa.append(habitat.load_file('ground_force', scene_number, flight_number))
            z.append(habitat.load_file('surface_dist', scene_number, flight_number))
    z = np.concatenate(z, axis=0)[:, 0]
    fa = np.concatenate(fa, axis=0)[:, 2]
    print(f'Mean {np.mean(fa)} | std {np.std(fa)}')

    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})

    z_reduced = z[np.where(z < 1.5)]

    fig, ax = plt.subplots(1, 1)
    ax.hist(z_reduced, bins='auto')
    ax.set_xlabel('Surface distance (m)')
    # ax.set_xlim([0, 1.5])
    # ax.set_ylabel('')
    ax.grid()
    plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/surface_dist_distribution.pdf')

    fig, ax = plt.subplots(1, 1)
    ax.hist(fa, bins=100)
    ax.set_xlabel('Ground effect (N)')
    x = 0.004
    # ax.set_xlim([-3e-2 * x, x])
    # ax.set_ylabel('')
    ax.grid()
    plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/fa_distribution_all.pdf')

    plt.hist(fa, bins='auto', density=True)


def plot_model_prediction(variant: str):
    model_d_path = f'{habitat.WD_PATH}analysis/habitat_residual_training/VariantD/20220215_165522/'
    model_d_point = f'{habitat.WD_PATH}analysis/habitat_residual_training/VariantD/20220215_165522/' \
                    f'checkpoints/epoch=20-step=1028.ckpt'
    model_a_path = f'{habitat.WD_PATH}analysis/habitat_residual_training/VariantA/20220211_203458/'

    drone_width, drone_height = 0.1, 0.025
    volumetric_mass = 200
    g = 9.81
    m = volumetric_mass * drone_width ** 2 * drone_height
    drone_prms = {'width': drone_width,  # [m]
                  'height': drone_height,  # [m]
                  'm': m,  # [kg]
                  'kt': m * g / (4 * 150 ** 2),  #
                  'km': m * g / (4 * 150 ** 2) * 2 ** 0.5 * drone_width / 2,  #
                  'l': 2 ** 0.5 * drone_width / 2,  # [m]
                  'ix': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iy': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iz': 1 / 3 * (drone_width ** 2 + drone_width ** 2) * m,  # [kg.m2]
                  'prop_radius': drone_width * 0.1}  # [m]
    theta = np.array([drone_prms['kt'] / drone_prms['m'],
                      drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                      drone_prms['km'] / drone_prms['iz'],
                      drone_prms['iz'] / drone_prms['ix']])

    uav_model = NumpyDynamicalModel({'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=np.float32, g=9.81)

    h_prms_d = pickle.load(open(f'{model_d_path}h_prms.pkl', 'rb'))
    model_d = VariantD.load_from_checkpoint(model_d_point,
                                            activation=h_prms_d['activation'],
                                            embedding=h_prms_d['embedding'],
                                            learning_rate=h_prms_d['lr'],
                                            uav_model=uav_model,
                                            motor_size=h_prms_d['motor_size'],
                                            past=h_prms_d['past'])
    model_d.eval()

    # h_prms_a = pickle.load(open(f'{model_a_path}h_prms.pkl', 'rb'))
    # model_a = VariantA.load_from_checkpoint(f'{model_a_path}default/20220211_203458/checkpoints/
    # epoch=1-step=241457.ckpt',
    #                                         activation=h_prms_a['activation'],
    #                                         learning_rate=h_prms_a['lr'],
    #                                         uav_model=uav_model,
    #                                         motor_size=h_prms_a['motor_size'],
    #                                         past=h_prms_a['past'])
    # model_a.eval()

    index = habitat.build_index_valid_flight()
    for scene_number, flight in index.items():
        for flight_number in flight:
            state = torch.from_numpy(habitat.load_file('state', scene_number, flight_number).copy())
            size = len(state)
            fa = torch.from_numpy(habitat.load_file('ground_force', scene_number, flight_number).copy())
            fa = fa[h_prms_d['past']-1:-1, 2]
            motor = torch.from_numpy(habitat.load_file('motor', scene_number, flight_number).copy())
            depth = torch.from_numpy(habitat.load_file('depth', scene_number, flight_number).copy())
            depth_emb = torch.from_numpy(habitat.load_file('depth_emb', scene_number, flight_number).copy())

            with torch.no_grad():
                fa_hat_d = []
                for i in range(h_prms_d['past'], len(state) - 1):
                    output = model_d.forward(motor[i], state[i], depth_emb[i - h_prms_d['past']:i])

                    fa_hat_d.append(output * habitat.HabitatResidualDataset.fa_std +
                                    habitat.HabitatResidualDataset.fa_mean)
                fa_hat_d = torch.stack(fa_hat_d, 0).squeeze()
            print(f'Error model d {((fa[0] - fa_hat_d[0]) ** 2).mean()}')
            # with torch.no_grad():
            #     fa_hat_a = []
            #     for i in tqdm(range(h_prms_a['past'], len(state) - 1), desc='Model a'):
            #         fa_hat_a.append(model_a.forward(motor[i], state[i], depth[i - h_prms_a['past']:i]))
            #         print()
            #     fa_hat_a = torch.stack(fa_hat_a, 0).squeeze()
            #     print(f'Error model a {((fa - fa_hat_a) ** 2).mean()}')

            dataset = habitat.HabitatResidualDataset('train', 40, False, True, False)
            sample = dataset[0]

            fig, ax = plt.subplots(1, 1)
            ax.plot(fa, label='Ground truth', marker='x')
            ax.plot(fa_hat_d, label='model d', marker='x')
            # ax.plot(fa_hat_a, label='model a')
            ax.grid()
            ax.set_ylabel('Ground force (N)')  # Surface distance (m)
            ax.set_xlabel('Sample number')
            ax.legend()
            plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/model_prediction.pdf')
            plt.show()


def main():
    # plot_fa_generator()
    plot_fa()
    # plot_model_prediction('d')


if __name__ == '__main__':
    main()
