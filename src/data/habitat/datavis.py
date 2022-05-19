import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from src.utils.transform import Transform as Tf
import src.data.habitat.loading as habitat
import src.data.euroc_mav.loading2 as euroc_mav
import src.data.habitat.data_gen as data_gen
import src.data.habitat.hatbitat2flu as hatbitat2flu
from src.drone_models.analytical_models.quadrotor_4parms import DynamicalModel
import src.utils.path as path
from src.utils.profiler import profiler
import torchvision.transforms.functional as f

plt.style.use('seaborn-deep')
plt.rcParams.update({"text.usetex": False,
                     "font.family": "serif",
                     "font.sans-serif": ["Computer Modern Roman"]})


def plot_fa_law():
    drone_prms = habitat.DRONE_PRMS

    def _apply(_z, _motor, _mu, _cap, coef):
        g = 9.81
        if _z <= np.sqrt(_mu) * drone_prms['prop_radius'] / 4:
            ground_force = _cap * drone_prms['m'] * g
        else:
            mu_r_z = _mu * (drone_prms['prop_radius'] / (4 * _z)) ** 2
            ground_force = np.clip(coef * np.sum(_motor ** 2) * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                   a_min=0, a_max=_cap * drone_prms['m'] * g)
        return ground_force

    motor = np.array([150, 150, 150, 150])
    z = np.linspace(1e-8, 0.8, 1000)
    fig, ax = plt.subplots(1, 1)

    from matplotlib.widgets import Slider
    from matplotlib.lines import Line2D

    mu = 20
    cap = 0.1
    coef = 10

    to_plot = np.array([_apply(e, motor, 1, 0.1, 1) for e in z])
    line: List[Line2D] = ax.plot(z, to_plot, label=rf'Before')
    to_plot = np.array([_apply(e, motor, mu, cap, coef) for e in z])
    line: List[Line2D] = ax.plot(z, to_plot, label='After')
    ax.set_title('Ground effect')
    ax.grid()
    ax.set_ylabel('Ground effect (N)')
    ax.set_xlabel('Surface distance (m)')
    ax.legend()

    plt.subplots_adjust(left=0.1, bottom=0.19, right=0.99, top=0.95, wspace=0, hspace=0)

    ax_mu = plt.axes([0.15, 0.06, 0.65, 0.03])
    mu_slider = Slider(ax=ax_mu, label="mu", valmin=0.01, valmax=20, valinit=mu, orientation="horizontal")
    ax_cap = plt.axes([0.15, 0.01, 0.65, 0.03])
    cap_slider = Slider(ax=ax_cap, label="cap", valmin=0, valmax=1, valinit=cap, orientation="horizontal")

    def update(_):
        mu_val = mu_slider.val
        cap_val = cap_slider.val
        _to_plot = np.array([_apply(e, motor, mu_val, cap_val, coef) for e in z])
        line[0].set_data([z, _to_plot])
        # ax.set_ylim([-0.003, np.max(_to_plot)])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

    mu_slider.on_changed(update)
    cap_slider.on_changed(update)
    # plt.tight_layout()
    plt.savefig(f'{habitat.WD_PATH}reports/improve/fa_generator.png')
    plt.show()


def plot_fa():
    index = habitat.build_index_valid_flight(False)

    fa = []
    z = []
    z_split = {'train': [], 'valid': [], 'test': []}
    zero_index = {scene_number: {} for scene_number in index}
    for scene_number, flight in index.items():
        if scene_number in habitat.TRAIN:
            include = 'train'
        elif scene_number in habitat.VALID:
            include = 'valid'
        else:
            assert scene_number in habitat.TEST
            include = 'test'
        for flight_number in flight:
            fa.append(habitat.load_file('ground_force', scene_number, flight_number))
            little_z = habitat.load_file('surface_dist', scene_number, flight_number)
            if (little_z == 0).any():
                print(f'{(little_z == 0).sum()} zero found in scene_number: {scene_number} | '
                      f'flight_number {flight_number}')
                zero_index[scene_number][flight_number] = (little_z == 0).sum()
            z_split[include].append(little_z)
            z.append(little_z)
    for key in z_split:
        z_split[key] = np.concatenate(z_split[key], 0)[:, 0]
    z = np.concatenate(z, axis=0)[:, 0]
    fa = np.concatenate(fa, axis=0)[:, 2]

    print(f'''z train: Mean {np.mean(z_split['train'])} | std {np.std(z_split['train'])}''')
    print(f'''z valid: Mean {np.mean(z_split['valid'])} | std {np.std(z_split['valid'])}''')
    print(f'''z test: Mean {np.mean(z_split['test'])} | std {np.std(z_split['test'])}''')

    print(f'z: Mean {np.mean(z)} | std {np.std(z)}')
    print(f'fa: Mean {np.mean(fa)} | std {np.std(fa)}')

    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})

    fig, ax = plt.subplots(1, 1)
    ax.hist(z, bins='auto')
    ax.set_xlabel('Surface distance (m)')
    ax.set_xlim([0, 1.5])
    # ax.set_ylabel('')
    ax.grid()
    plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/surface_dist_distribution.pdf')

    fig, ax = plt.subplots(1, 1)
    ax.hist(fa, bins='auto')
    ax.set_xlabel('Ground effect (N)')
    x = 0.004
    ax.set_xlim([0, x])
    # ax.set_ylabel('')
    ax.grid()
    plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/fa_distribution_all.pdf')

    fig, ax = plt.subplots(1, 1)
    ax.plot(z)
    ax.set_xlabel('Sample #')
    ax.set_ylabel('Surface distance (m)')
    ax.grid()


def plot_model_prediction(name):
    msg = ['nogru', 'all', 'emb', 'fine']
    assert name in msg, f'Name must be in {msg}'
    use_cuda_if_possible = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')

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
    uav_model = DynamicalModel({'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=np.float32, g=9.81)

    from src.drone_models.residual.all_height_no_gru import AllHeightNoGRU as Model
    path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220224_095722/'
    model_name = 'epoch=28-step=18936-val_loss=0.57.ckpt'
    h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
    model_nogru = Model.load_from_checkpoint(f'{path}{model_name}',
                                             activation=h_prms['activation'],
                                             l2=h_prms['l2'],
                                             learning_rate=h_prms['lr'],
                                             uav_model=uav_model,
                                             motor_size=h_prms['motor_size'],
                                             past=h_prms['past']).to(device)

    if name == 'all':
        from src.drone_models.residual.all_height import AllHeight as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220223_170234/'
        model_name = 'epoch=21-step=14365-val_loss=0.51.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)

    elif name == 'emb':
        from src.drone_models.residual.emb_height import EmbHeight as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220223_121251/'
        model_name = 'epoch=136-step=13425-val_loss=0.50.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           embedding=h_prms['embedding'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
    else:
        assert name == 'fine'
        from src.drone_models.residual.fine_tuning import FineTunning as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220224_183259/'
        model_name = 'epoch=0-step=49005-val_loss=0.49.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)

    model.eval()

    def _apply(_z, _motor):
        mu = 1
        cap = 0.1
        ground_force = np.zeros(3)
        if _z < np.sqrt(mu) * drone_prms['prop_radius'] / 4:
            ground_force[2] = cap * drone_prms['m'] * g
        else:
            mu_r_z = mu * (drone_prms['prop_radius'] / (4 * _z)) ** 2
            ground_force[2] = np.clip(np.sum(_motor ** 2) * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                      a_min=0, a_max=cap * drone_prms['m'] * g)
        return ground_force

    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std

    index = habitat.build_index_valid_flight(True)
    for scene_number in tqdm(habitat.VALID, desc='scene'):
        scene_number = 64
        flight = index[scene_number]
        for flight_number in tqdm(flight, desc='flight'):
            flight_number = 14
            state = torch.from_numpy(habitat.load_file('state', scene_number, flight_number).copy()).to(device)
            fa = torch.from_numpy(habitat.load_file('ground_force', scene_number, flight_number).copy()).to(device)
            z = torch.from_numpy(habitat.load_file('surface_dist', scene_number, flight_number).copy())
            motor = torch.from_numpy(habitat.load_file('motor', scene_number, flight_number).copy())
            depth_emb = torch.from_numpy(habitat.load_file('depth_emb', scene_number, flight_number).copy()).to(device)
            depth = torch.from_numpy(habitat.load_file('depth', scene_number, flight_number).copy()).to(device)

            assert len(fa) == len(motor)
            assert len(fa) + 1 == len(z)
            assert len(fa) + 1 == len(depth_emb)

            # fa = fa[h_prms['past'] - 1:-1, 2]
            xx = torch.arange(h_prms['past'] - 1, len(fa))
            fa_gt = fa[h_prms['past'] - 1:, 2]
            motor = motor[h_prms['past'] - 1:].numpy()
            z_gt = z[h_prms['past'] - 1:-1, 0]
            assert len(xx) == len(fa_gt)
            assert len(fa_gt) == len(z_gt)

            with torch.no_grad():
                # fa_hat = torch.empty(len(fa), dtype=torch.float32) * torch.nan
                z_hat = torch.empty(len(z_gt), dtype=torch.float32)
                z_hat2 = torch.empty(len(z_gt), dtype=torch.float32)
                for i in tqdm(range(len(z_hat)), desc='prediction'):
                    output = model(depth_emb[i:i + h_prms['past']]).cpu()
                    z_hat[i] = (output * z_std + z_mean)
                    output2 = model_nogru(depth[i + h_prms['past'] - 1].reshape(1, 480, 640)).cpu()
                    z_hat2[i] = (output2 * z_std + z_mean)
            print(f'{scene_number} | {flight_number} | '
                  f'Error loss wandb emb_height {((((z_hat - z) - z_mean) / z_std) ** 2).mean():0.4f} | '
                  f'Real loss {((z_hat - z) ** 2).mean():0.4f}')

            fa_gt = [_apply(e, motor)[2] for e in z_gt.numpy()]
            fa_hat = [_apply(e, motor)[2] for e in z_hat.numpy()]
            fa_hat2 = [_apply(e, motor)[2] for e in z_hat2.numpy()]

            fig, ax = plt.subplots(1, 1)
            ax.set_title(f'Scene number: {scene_number} | flight_number: {flight_number}')
            ax.plot(xx, z_gt, label='Ground truth', marker=',')
            ax.plot(xx, z_hat, label='Ours', marker=',')
            ax.plot(xx, z_hat2, label='No gru', marker=',')
            # ax.plot(fa_hat_a, label='model a')
            ax.grid()
            ax.set_ylabel('z (m)')  # Surface distance (m)
            ax.set_xlabel('Sample number')
            ax.legend()
            plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/test_emb_depth/'
                        f'model_prediction-{scene_number}-{flight_number}.pdf')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.set_title(f'Scene number: {scene_number} | flight_number: {flight_number}')
            ax.plot(xx, fa_gt, label='Ground truth', marker=',')
            ax.plot(xx, fa_hat, label='Ours', marker=',')
            ax.plot(xx, fa_hat2, label='No gru', marker=',')
            # ax.plot(fa_hat_a, label='model a')
            ax.grid()
            ax.set_ylabel('Fa (N)')  # Surface distance (m)
            ax.set_xlabel('Sample number')
            ax.legend()
            # plt.show()
            plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/test_emb_depth/'
                        f'fa_model_prediction-{scene_number}-{flight_number}.pdf')
            plt.close(fig)
            pickle.dump(z_gt, open(f'{habitat.WD_PATH}figure6/z_gt.pkl', 'wb'))
            pickle.dump(z_hat, open(f'{habitat.WD_PATH}figure6/z_hat.pkl', 'wb'))
            pickle.dump(z_hat2, open(f'{habitat.WD_PATH}figure6/z_hat2.pkl', 'wb'))
            pickle.dump(fa_gt, open(f'{habitat.WD_PATH}figure6/fa_gt.pkl', 'wb'))
            pickle.dump(fa_hat, open(f'{habitat.WD_PATH}figure6/fa_hat.pkl', 'wb'))
            pickle.dump(fa_hat2, open(f'{habitat.WD_PATH}figure6/fa_hat2.pkl', 'wb'))
            return
    return
    cluster_scene_path = f'/home/qpossamai/phd/turbulence/figure6/'
    scene_path = f'~/phd/turbulence/figure6/'
    cluster_scene_path = f'/home/qpossamai/phd/turbulence/reports/IROS_01_03_2022/graphics/test_emb_depth'
    scene_path = f'~/phd/turbulence/reports/IROS_01_03_2022/graphics/test_emb_depth/'
    command = f'rsync -rtv --progress remsto:{cluster_scene_path} {scene_path}'
    if not os.path.isfile(scene_path):
        print(command)
        raise (FileNotFoundError('Scene is not in local'))
        os.system(command)  # if scene not downloaded

    scene_path = '/Users/quentin/phd/turbulence/data/habitat/scene_datasets/mp3d/TbHJrupSAjP/flight2/7/depth.npy'
    cluster_scene_path = f'/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/TbHJrupSAjP/flight2/7/depth.npy'
    command = f'rsync -rtv --progress remsto:{cluster_scene_path} {scene_path}'
    print(command)
    os.system(command)  # if scene not downloaded


def rmse_model_prediction(name, dataset_name: str):
    msg = ['habitat', 'euroc_mav']
    assert dataset_name in msg, f'dataset must be in {msg}'
    use_cuda_if_possible = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')

    msg = ['emb_norm', 'nogru', 'all', 'emb', 'fine', 'fine_norm']
    assert name in msg, f'Name must be in {msg}'

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
    uav_model = DynamicalModel({'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=np.float32, g=9.81)

    if name == 'nogru':
        from src.drone_models.residual.all_height_no_gru import AllHeightNoGRU as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220224_095722/'
        model_name = 'epoch=28-step=18936-val_loss=0.57.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, False)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, False)

    elif name == 'all':
        from src.drone_models.residual.all_height import AllHeight as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220223_170234/'
        model_name = 'epoch=21-step=14365-val_loss=0.51.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, False)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, False)

    elif name == 'emb':
        from src.drone_models.residual.emb_height import EmbHeight as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220223_121251/'
        model_name = 'epoch=136-step=13425-val_loss=0.50.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           embedding=h_prms['embedding'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
        if dataset_name == 'euroc_mav':
            print(f'name: {name} | dataset_name: {dataset_name} | Emb not generated')
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], False, True, False, False)

    elif name == 'emb_norm':
        from src.drone_models.residual.emb_height2 import EmbHeight as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220227_200520/'
        model_name = 'epoch=90-step=8917-val_loss=0.54.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           embedding=h_prms['embedding'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
        if dataset_name == 'euroc_mav':
            print(f'name: {name} | dataset_name: {dataset_name} | Emb not generated')
            return
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], False, False, True, False)
    elif name == 'fine':
        # from src.drone_models.residual.fine_tuning2 import FineTunning as Model
        # path = f'{habitat.WD_PATH}analysis/habitat_residual_training/FineTunning/20220224_183259/'
        # model_name = 'epoch=0-step=49005-val_loss=0.49.ckpt'
        from src.drone_models.residual.fine_tuning import FineTunning as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/FineTunning/20220226_183121/'
        model_name = 'epoch=2-step=147017-val_loss=0.48.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, True)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, True)

    else:
        assert name == 'fine_norm'
        # from src.drone_models.residual.fine_tuning2 import FineTunning as Model
        # path = f'{habitat.WD_PATH}analysis/habitat_residual_training/FineTunning/20220224_183259/'
        # model_name = 'epoch=0-step=49005-val_loss=0.49.ckpt'
        from src.drone_models.residual.fine_tuning2 import FineTunning as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/FineTunning/20220228_092516/'
        model_name = 'epoch=0-step=6124-val_loss=0.51.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past']).to(device)
        if dataset_name == 'euroc_mav':
            dataset = euroc_mav.EurocMAVDatasetHorizon2('test', h_prms['past'], True, False, False, True)
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], True, False, False, True)

    if dataset_name == 'habitat':
        def compute_ground_effect(motor: np.ndarray, z: float, mu=1, cap=0.1) -> np.ndarray:
            g = 9.81
            ground_force = np.zeros(3)
            if z <= np.sqrt(mu) * drone_prms['prop_radius'] / 4:
                ground_force[2] = cap * drone_prms['m'] * g
            else:
                mu_r_z = mu * (drone_prms['prop_radius'] / (4 * z)) ** 2
                ground_force[2] = np.clip(np.sum(motor ** 2) * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                          a_min=0, a_max=cap * drone_prms['m'] * g)
            return np.concatenate([ground_force, np.zeros(3)]).squeeze()[2]
    else:
        assert dataset_name == 'euroc_mav'

        def compute_ground_effect(motor: np.ndarray, z: float) -> np.ndarray:
            sol = [9.40693193e-07, -4.17981623e-03]
            return sol[0] * (motor ** 2).sum(axis=-1) / (z ** 2 - sol[1])

    data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True, drop_last=False)

    print(f'past: {h_prms["past"]}')

    model.eval()

    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std
    fa_mean, fa_std = habitat.HabitatResidualDataset.fa_mean, habitat.HabitatResidualDataset.fa_std

    z_loss = (np.empty(len(data), dtype=np.float32) * np.nan)
    fa_loss = (np.empty(len(data), dtype=np.float32) * np.nan)
    z_hat_log = (np.empty(len(data), dtype=np.float32) * np.nan)
    fa_hat_log = (np.empty(len(data), dtype=np.float32) * np.nan)
    z_gt_log = (np.empty(len(data), dtype=np.float32) * np.nan)
    fa_gt_log = (np.empty(len(data), dtype=np.float32) * np.nan)
    fa_gttest_log = (np.empty(len(data), dtype=np.float32) * np.nan)
    aquoiilsertcepoint = []
    print(name)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data), total=len(data)):
            if name == 'emb':
                depth = batch['depth_emb'].to(device)
            elif name == 'emb_norm':
                depth = batch['norm_emb'].to(device)
            else:
                depth = batch['depth'].to(device)
            motor = batch['motor'].cpu().numpy()

            z_hat = model(depth).cpu().numpy() * z_std + z_mean
            z_gt = batch['surface_dist'].cpu().numpy() * z_std + z_mean
            loss = ((z_gt - z_hat) ** 2).squeeze()
            z_loss[i] = loss
            z_hat_log[i] = z_hat.squeeze()
            z_gt_log[i] = z_gt.squeeze()

            fa_gt = batch['ground_force'].cpu().numpy().squeeze()[2] * fa_std + fa_mean
            fa_gt_test = compute_ground_effect(motor, z_gt)
            # if dataset_name == 'habitat':
            # assert np.isclose(fa_gt, fa_gt_test, atol=1e-8).all()
            fa_hat = compute_ground_effect(motor, z_hat)
            loss = (fa_gt_test - fa_hat) ** 2
            fa_loss[i] = loss
            fa_hat_log[i] = fa_hat
            fa_gt_log[i] = fa_gt
            fa_gttest_log[i] = fa_gt_test
            aquoiilsertcepoint.append(batch['flight'])

    if dataset_name == 'euroc_mav':
        pickle.dump(aquoiilsertcepoint, open(f'{habitat.WD_PATH}aquoiilsertcepoint_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(z_hat_log, open(f'{habitat.WD_PATH}z_hat_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(fa_hat_log, open(f'{habitat.WD_PATH}fa_hat_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(z_gt_log, open(f'{habitat.WD_PATH}z_gt_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(fa_gt_log, open(f'{habitat.WD_PATH}fa_gt_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(fa_gttest_log, open(f'{habitat.WD_PATH}fa_gttest_{name}_{dataset_name}.pkl', 'wb'))

    fig, ax = plt.subplots(1, 1)
    ax.plot(z_gt_log)
    ax.plot(z_hat_log)
    ax.legend()
    plt.savefig(f'{habitat.WD_PATH}z_{name}_{dataset_name}.png')

    fig, ax = plt.subplots(1, 1)
    ax.plot(fa_hat_log)
    ax.plot(fa_gt_log)
    ax.plot(fa_gttest_log)
    ax.legend()
    plt.savefig(f'{habitat.WD_PATH}fa_{name}_{dataset_name}.png')
    print(f'rmse z_loss: {np.sqrt(z_loss.mean())}')
    print(f'rmse fa_loss: {np.sqrt(fa_loss.mean())}')
    print(f'\n\n')


def observe_generated_traj(scene_number, flight_number):
    print(f'Observing generated trajectory scene number {scene_number} and flight number {flight_number}')
    labels = pickle.load(open(f'/Users/quentin/phd/turbulence/data/habitat/labels.pkl', 'rb'))

    try:  # check if trajectory is labelled
        assert not np.isnan(labels[scene_number][flight_number]).any()
        return
    except KeyError:  # if it does not exist, create it
        if flight_number != 1:
            pass
        else:  # flight_number == 1
            try:  # if 0 exist, 1 is the same (why I don't know, maybe seed in N*)
                state1 = habitat.load_file('state', scene_number, flight_number)
                state0 = habitat.load_file('state', scene_number, 0)
                assert (state0.shape == state1.shape)
                assert (state0 == state1).all()
                labels[scene_number][flight_number] = labels[scene_number][0]
                pickle.dump(labels, open(f'/Users/quentin/phd/turbulence/data/habitat/labels.pkl', 'wb'))
                return
            except (FileNotFoundError, AssertionError):  # If 0 does not exist, create 1
                pass

    scene_id = habitat.MP3D_SCENE_LIST[scene_number]
    scene_path = f"{habitat.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    cluster_scene_path = f"/storage/Datasets/3D-Simulator-Datasets/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    traj_path = f"{habitat.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{habitat.FLIGHT}/{flight_number}/"
    state = habitat.load_file('state', scene_number, flight_number)
    state_habitat = hatbitat2flu.Habitat2FLU.state_flu_2_state_habitat(state)
    tf_follow = Tf.from_quat_trans(state_habitat[:, 3:7], state_habitat[:, :3]).as_matrix()

    # If scene is not downloaded
    command = f'rsync -tvP --progress remsto:{cluster_scene_path} {scene_path}'
    if not os.path.isfile(scene_path):
        print(command)
        raise (FileNotFoundError('Scene is not in local'))
        os.system(command)  # if scene not downloaded

    simulation = data_gen.Simulation(scene_path, camera_resolution=np.array([480, 640]),
                                     drone_prms=habitat.DRONE_PRMS,
                                     gravity=9.81)
    label = simulation.interactive_sim(camera_rgb=True, camera_depth=True, tf_follow=tf_follow, return_label=True)
    labels[scene_number][flight_number] = label

    pickle.dump(labels, open(f'/Users/quentin/phd/turbulence/data/habitat/labels.pkl', 'wb'))
    simulation.sim.close(True)


def observe_traj_pred(scene_number, flight_number):
    model_path = f'{path.ANALYSIS_PATH}habitat_residual_training/'
    model_part_path = 'AllHeight/20220513_154038/'
    model_ckpt = 'epoch=78-step=894042-val_loss=0.04.ckpt'
    model_path = f'{model_path}{model_part_path}'
    name = f'''{model_part_path.replace('/', '_')}{model_ckpt[:-5]}'''
    past = 1

    color = {0: 'white', 1: 'red', 2: 'green', 3: 'yellow'}

    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    if scene_number in habitat.TRAIN:
        split = 'Train'
    elif scene_number in habitat.VALID:
        split = 'Valid'
    else:
        assert scene_number in habitat.TEST
        split = 'Test'

    scene_id = habitat.MP3D_SCENE_LIST[scene_number]
    scene_path = f"{path.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    cluster_scene_path = f"/storage/Datasets/3D-Simulator-Datasets/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    sim = data_gen.Simulation(scene_path, camera_resolution=np.array([480, 640]),
                              drone_prms=habitat.DRONE_PRMS,
                              gravity=9.81)
    label = labels[scene_number][flight_number]

    flight_folder = f'{habitat.get_flight_folder(scene_number, flight_number)}pred/'
    pred = np.load(f'{flight_folder}{name}.npz')

    loss = pred['loss']
    z_gt = pred['z_gt']
    z_hat = pred['z_hat']

    state = habitat.load_file('state', scene_number, flight_number)
    state_habitat = hatbitat2flu.Habitat2FLU.state_flu_2_state_habitat(state)
    tf_follow = Tf.from_quat_trans(state_habitat[:, 3:7], state_habitat[:, :3]).as_matrix()

    # If scene is not downloaded
    command = f'rsync -tvP --progress remsto:{cluster_scene_path} {scene_path}'
    if not os.path.isfile(scene_path):
        print(command)
        raise (FileNotFoundError('Scene is not in local'))
        os.system(command)  # if scene not downloaded

    from matplotlib.lines import Line2D
    import matplotlib.image as mpl_image
    from matplotlib.text import Text
    # Interactive plot
    iteration = [0]
    sim.agent.transformation = tf_follow[iteration[0]]
    obs = sim.sim.get_sensor_observations(agent_ids=0)
    im = [obs['left_camera']]
    depth = [obs['left_depth_camera']]
    z = [obs['surface_dist'][0]]

    image_mode = [False]
    vmax = None

    fig, (ax_im, ax_z) = plt.subplots(1, 2, figsize=(12.8, 8))
    fig: plt.Figure
    fig.suptitle(f'Scene {scene_number} | flight {flight_number} | split: {split}', fontsize=14)

    ax_im: plt.Axes
    ax_z: plt.Axes

    if image_mode[0]:
        line_im: List[mpl_image.AxesImage] = [ax_im.imshow(im[0])]
    else:
        line_im: List[mpl_image.AxesImage] = [ax_im.imshow(depth[0], vmax=vmax)]
    line_z_gt: Line2D = ax_z.plot(z_gt, label='z GT')
    line_z_hat: Line2D = ax_z.plot(range(past - 1, len(z_gt)), z_hat, label='z hat')
    line_v: Line2D = ax_z.axvline(iteration[0], c='k')

    slices = np.concatenate([np.zeros(1), np.where(np.diff(label) != 0)[0] + 2, np.array([len(label)])])
    for i in range(len(slices) - 1):
        start = int(slices[i])
        end = int(slices[i + 1]) - 1
        print(start, end)
        ax_z.axvspan(start - 0.5, end + 0.5, color=color[label[start]], alpha=0.5)

    ax_z.legend()
    ax_z.grid()
    # ax_z.set_ylabel('z')
    ax_z.set_xlabel('Time (sample #)')
    title: Text = ax_im.set_title(f'Frame # {iteration[0]}')

    def update_plot():
        print(f'Frame #: {iteration[0]}')
        sim.agent.transformation = tf_follow[iteration[0]]
        # Replace camera
        s = sim.sim.get_agent(0).state
        s.sensor_states['surface_dist'].rotation = np.quaternion(np.cos(0.25 * np.pi),
                                                                 -np.sin(0.25 * np.pi), 0, 0)
        sim.sim.get_agent(0).set_state(s, False, False)

        obs = sim.sim.get_sensor_observations(agent_ids=0)
        im[0] = obs['left_camera']
        depth[0] = obs['left_depth_camera']
        z[0] = obs['surface_dist'][0]
        if image_mode[0]:
            line_im[0].set_data(im[0])
        else:
            line_im[0].set_data(depth[0])
        ax_z.set_xlim([iteration[0] - 40, iteration[0] + 40])
        line_v.set_data(([iteration[0], iteration[0]], [0, 1]))

        # ax_im.set_title(f'Frame # {iteration[0]}')
        title.set_text(f'Frame # {iteration[0]}')
        fig.canvas.draw()

    def on_press(event):
        # print('press', event.key)
        if event.key == 'v':
            iteration[0] = iteration[0] + 1
            update_plot()
        if event.key == 'c':
            iteration[0] = iteration[0] - 1
            update_plot()
        if event.key == 'x':
            ax_z.set_xlim(0, len(z_gt))
            fig.canvas.draw()
        if event.key == 'n':
            image_mode[0] = not image_mode[0]
            if image_mode[0]:
                line_im[0] = ax_im.imshow(im[0])
                # line_im[0].set_data(im[0])
            else:
                line_im[0] = ax_im.imshow(depth[0], vmax=vmax)
                # line_im[0].set_data(depth[0])
            fig.canvas.draw()

    def on_click(event):
        # print('press', event.key)
        if event.button == 1:  # 1 for left click, 3 for right click,
            iteration[0] = np.round(event.xdata).astype(int)
            update_plot()

    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # plt.tight_layout()

    plt.show()
    sim.sim.close(True)


def boxplot_pred():
    dataset = habitat.HabitatResidualDataset('test', 40, False, False, False, False)
    z_mean, z_std = dataset.z_mean, dataset.z_std
    fa_mean, fa_std = dataset.fa_mean, dataset.fa_std
    dataset_name = 'habitat'  # habitat  euroc_mav
    z_gt = pickle.load(open(f'{habitat.WD_PATH}data_prediction/z_gt_fine_{dataset_name}.pkl', 'rb'))
    fa_gt = pickle.load(open(f'{habitat.WD_PATH}data_prediction/fa_gt_fine_{dataset_name}.pkl', 'rb'))
    z_hat = pickle.load(open(f'{habitat.WD_PATH}data_prediction/z_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_hat = pickle.load(open(f'{habitat.WD_PATH}data_prediction/fa_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_gttest = pickle.load(open(f'{habitat.WD_PATH}data_prediction/fa_gttest_fine_{dataset_name}.pkl', 'rb'))

    fa_or_z = 'z'
    if fa_or_z == 'z':
        hat = z_hat
        gt = z_gt
        plt.plot(gt, label=f'{fa_or_z} GT')
        plt.plot(hat, label=f'{fa_or_z} (Ours)')
        plt.ylabel(f'Height (m)')
        plt.xlabel(f'Sample #')
        plt.legend()
        plt.scatter(z_gt, fa_gt, s=0.1)
        plt.scatter(z_hat, fa_hat, s=0.1)

    else:
        hat = fa_hat
        gt = fa_gt

        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(gt, label=f'{fa_or_z} GT')
        ax.plot(fa_gttest, label=f'{fa_or_z} (Analytical ground effect from GT heights)')
        ax.plot(hat, label=f'{fa_or_z} (Ours)')
        ax.set_ylabel(f'Ground Effect (N)')
        ax.set_xlabel(f'Sample #')
        ax.set_facecolor('w')
        ax.legend()
        # ax.grid(True)
        plt.subplots_adjust(left=0.11, bottom=0.1, right=0.99, top=0.99, wspace=0.3, hspace=0.3)
        plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/fa_euroc_prediction.pdf')

        np.sqrt(((gt - fa_gttest) ** 2).mean())
        np.sqrt(((fa_gttest - hat) ** 2).mean())
        np.sqrt(((gt - hat) ** 2).mean())

    hist1, bin_edges1 = np.histogram(gt, bins=200)
    hist2 = np.zeros(len(hist1))
    box_plot = []
    for i in range(len(bin_edges1) - 2):
        box_plot.append(hat[np.where(np.logical_and(bin_edges1[i] <= gt, gt < bin_edges1[i + 1]))[0]])
        hist2[i] = len(box_plot[i])
    i = len(bin_edges1) - 2
    box_plot.append(hat[np.where(np.logical_and(bin_edges1[i] <= gt, gt <= bin_edges1[i + 1]))[0]])
    hist2[i] = len(box_plot[i])
    diff_edges1 = np.diff(bin_edges1)

    fig, ax = plt.subplots()
    ax.bar(bin_edges1[:-1], hist1, width=diff_edges1, align="edge")
    ax.set_xlabel(f'fa GT (N)' if fa_or_z == 'fa' else f'z GT (m)' + f' {dataset_name}')
    if dataset_name == 'habitat':
        ax.set_xlim([0, 1.8])
    # ax.bar(bin_edges1[:-1], hist2, width=diff_edges1, align="edge")
    ax.grid()

    fig, ax = plt.subplots()
    positions = bin_edges1[:-1] + diff_edges1 / 2
    ax.boxplot(box_plot, showfliers=False)
    initial_ticks = ax.get_xticks()
    ax.plot(initial_ticks, positions, label='y=x')
    ax.set_xlabel(f'fa GT (N)' if fa_or_z == 'fa' else f'z GT (m)')
    ax.set_ylabel(f'fa (Ours) (N)' if fa_or_z == 'fa' else f'z (Ours) (m)')
    plt.xticks(initial_ticks, [f'{e:.3f}' for e in positions], rotation=-45)
    ax.legend()

    plt.plot(gt)
    plt.plot(hat)

    len(z_hat)


def euroc_mav_table():
    dataset_name = 'euroc_mav'
    z_gt = pickle.load(open(f'{habitat.WD_PATH}data_prediction/z_gt_fine_{dataset_name}.pkl', 'rb'))
    fa_gt = pickle.load(open(f'{habitat.WD_PATH}data_prediction/fa_gt_fine_{dataset_name}.pkl', 'rb'))
    z_hat = pickle.load(open(f'{habitat.WD_PATH}data_prediction/z_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_hat = pickle.load(open(f'{habitat.WD_PATH}data_prediction/fa_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_gttest = pickle.load(open(f'{habitat.WD_PATH}data_prediction/fa_gttest_fine_{dataset_name}.pkl', 'rb'))

    dataset = euroc_mav.EurocMAVDatasetHorizon2('test', 40, False, False, False, False)
    print(f'Balise 1')

    # flight = [dataset[i]['flight'] for i in tqdm(range(len(dataset)))]
    flight_path = f'{habitat.WD_PATH}data_prediction/flight.pkl'
    # pickle.dump(flight, open(flight_path, 'wb'))
    flight = pickle.load(open(flight_path, 'rb'))

    error_z = {"V1_01_easy/": [],
               "V1_02_medium/": [],
               "V1_03_difficult/": []}
    error_fa = {"V1_01_easy/": [],
                "V1_02_medium/": [],
                "V1_03_difficult/": []}
    error_fa_cheeseman = {"V1_01_easy/": [],
                          "V1_02_medium/": [],
                          "V1_03_difficult/": []}
    print(f'Balise 2')

    for i, _ in tqdm(enumerate(z_gt)):
        # assert z_gt[i] == dataset[i]['surface_dist'] * dataset.z_std + dataset.z_mean
        error_z[flight[i]].append((z_gt[i] - z_hat[i]) ** 2)
        error_fa[flight[i]].append((fa_gt[i] - fa_hat[i]) ** 2)
        error_fa_cheeseman[flight[i]].append((fa_gt[i] - fa_gttest[i]) ** 2)
    print(f'Balise 3')

    total_z = []
    total_fa = []
    total_fa_cheese = []
    for e in error_z:
        print(f'{e}')
        print(f'Error z {e}: {np.sqrt(np.mean(error_z[e]))}')
        print(f'Error fa {e}: {np.sqrt(np.mean(error_fa[e]))}')
        print(f'Error fa_cheese {e}: {np.sqrt(np.mean(error_fa_cheeseman[e]))}')
        print(f'\n\n')
        total_z = total_z + error_z[e]
        total_fa = total_fa + error_fa[e]
        total_fa_cheese = total_fa_cheese + error_fa_cheeseman[e]
    print(f'Error z total: {np.sqrt(np.mean(total_z))}')
    print(f'Error fa total: {np.sqrt(np.mean(total_fa))}')
    print(f'Error fa_cheese total: {np.sqrt(np.mean(total_fa_cheese))}')
    print()


def get_image_from_generated_traj(scene_number, flight_number, idx: list):
    scene_id = habitat.MP3D_SCENE_LIST[scene_number]
    scene_path = f"{habitat.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    cluster_scene_path = f"/storage/Datasets/3D-Simulator-Datasets/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    traj_path = f"{habitat.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{habitat.FLIGHT}/{flight_number}/"
    state = habitat.load_file('state', scene_number, flight_number)
    motor = habitat.load_file('motor', scene_number, flight_number)
    state_habitat = hatbitat2flu.Habitat2FLUACC.state_flu_2_state_habitat(state)
    tf_follow = Tf.from_quat_trans(state_habitat[:, 3:7], state_habitat[:, :3]).as_matrix()[idx]

    command = f'rsync -Ptv  remsto:{cluster_scene_path} {scene_path}'
    if not os.path.isfile(scene_path):
        print(command)
        raise (FileNotFoundError('Scene is not in local'))
        os.system(command)  # if scene not downloaded

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
    mu = 1
    cap = 0.1

    def _apply(_z, _motor):
        ground_force = np.zeros(3)
        if _z < np.sqrt(mu) * drone_prms['prop_radius'] / 4:
            ground_force[2] = cap * drone_prms['m'] * g
        else:
            mu_r_z = mu * (drone_prms['prop_radius'] / (4 * _z)) ** 2
            ground_force[2] = np.clip(np.sum(_motor ** 2) * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                      a_min=0, a_max=cap * drone_prms['m'] * g)
        return ground_force[2]

    simulation = data_gen.Simulation(scene_path, camera_resolution=[480, 640], drone_prms=drone_prms, gravity=9.81)
    # from src.drone_models.residual.emb_height2 import EmbHeight as Model
    # path = f'{habitat.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220227_200520/'
    # model_name = 'epoch=90-step=8917-val_loss=0.54.ckpt'
    from src.drone_models.residual.emb_height import EmbHeight as Model
    path = f'{habitat.WD_PATH}analysis/habitat_residual_training/EmbHeight/20220223_121251/'
    model_name = 'epoch=136-step=13425-val_loss=0.50.ckpt'
    #
    h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
    model = Model.load_from_checkpoint(f'{path}{model_name}',
                                       activation=h_prms['activation'],
                                       embedding=h_prms['embedding'],
                                       l2=h_prms['l2'],
                                       learning_rate=h_prms['lr'],
                                       uav_model=None,
                                       motor_size=h_prms['motor_size'],
                                       past=h_prms['past'])

    from src.drone_models.residual.all_height_no_gru import AllHeightNoGRU as Model
    path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220224_095722/'
    model_name = 'epoch=28-step=18936-val_loss=0.57.ckpt'
    h_prms_nogru = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
    model_nogru = Model.load_from_checkpoint(f'{path}{model_name}',
                                             activation=h_prms_nogru['activation'],
                                             l2=h_prms_nogru['l2'],
                                             learning_rate=h_prms_nogru['lr'],
                                             uav_model=None,
                                             motor_size=h_prms_nogru['motor_size'],
                                             past=h_prms_nogru['past'])

    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std

    rgb = []
    z = []
    fa = []
    z_hat = []
    fa_hat = []
    z_hat_nogru = []
    fa_hat_nogru = []
    motor = motor[idx]
    with torch.no_grad():
        for i, tf in enumerate(tqdm(tf_follow)):
            ind = idx[i]
            simulation.agent.transformation = tf
            obs = simulation.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
            rgb.append(obs['left_camera'])
            z.append(obs['surface_dist'])
            fa.append(_apply(obs['surface_dist'], motor[i]))
            depth_emb = habitat.load_file('depth_emb', scene_number, flight_number)[ind - h_prms['past'] + 1:ind + 1]
            depth_emb = torch.from_numpy(depth_emb.copy())
            z_hat.append(model(depth_emb).squeeze().item() * z_std + z_mean)
            fa_hat.append(_apply(z_hat[-1], motor[i]))
            depth = habitat.load_file('depth', scene_number, flight_number)[ind:ind + 1]
            depth = torch.from_numpy(depth.copy())
            z_hat_nogru.append(model_nogru(depth).squeeze().item() * z_std + z_mean)
            fa_hat_nogru.append(_apply(z_hat_nogru[-1], motor[i]))
    simulation.sim.close(True)

    return (np.stack(rgb, axis=0),
            np.stack(z, axis=0).flatten(),
            np.stack(fa, axis=0).flatten(),
            np.stack(z_hat, axis=0).flatten(),
            np.stack(fa_hat, axis=0).flatten(),
            np.stack(z_hat_nogru, axis=0).flatten(),
            np.stack(fa_hat_nogru, axis=0).flatten(),
            )


def predict_good_fa():
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
            ground_force[2] = np.clip(np.sum(_motor ** 2) * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                      a_min=0, a_max=cap * drone_prms['m'] * g)
        return ground_force

    motor = np.array([150, 150, 150, 150])
    dataset_name = 'habitat'  # habitat  euroc_mav

    z_gt = pickle.load(open(f'{habitat.WD_PATH}data_prediction/z_gt_fine_{dataset_name}.pkl', 'rb'))
    dataset = habitat.HabitatResidualDataset('test', 1, False, False, False, False)
    data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    z_gt_big = []
    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std
    fa_mean, fa_std = habitat.HabitatResidualDataset.fa_mean, habitat.HabitatResidualDataset.fa_std
    for i, batch in tqdm(enumerate(data), total=len(data)):
        z_gt_big.append(batch['surface_dist'].numpy() * z_std + z_mean)

    for e in [f'hat_all_{dataset_name}.pkl', f'hat_emb_{dataset_name}.pkl', f'hat_fine_{dataset_name}.pkl',
              f'hat_nogru_{dataset_name}.pkl']:
        z_hat = pickle.load(open(f'{habitat.WD_PATH}data_prediction/z_{e}', 'rb'))
        if 'all' in e or 'nogru' in e:
            fa_gt_corrected = np.array([_apply(e, motor) for e in z_gt_big])[:, 2]
        else:
            fa_gt_corrected = np.array([_apply(e, motor) for e in z_gt])[:, 2]
        fa_hat_corrected = np.array([_apply(e, motor) for e in z_hat])[:, 2]
        loss = np.sqrt(((fa_gt_corrected - fa_hat_corrected) ** 2).mean())
        print(f'fa{e}  Loss: {loss}')

        pickle.dump(fa_gt_corrected, open(f'{habitat.WD_PATH}data_prediction/fa_gt_corrected', 'wb'))
        pickle.dump(fa_hat_corrected, open(f'{habitat.WD_PATH}data_prediction/fa_{e}_corrected', 'wb'))


def analyse_prediction():
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    model_path = f'{path.ANALYSIS_PATH}habitat_residual_training/'
    model_part_path = 'AllHeight/20220513_154038/'
    model_ckpt = 'epoch=78-step=894042-val_loss=0.04.ckpt'
    model_path = f'{model_path}{model_part_path}'
    name = f'''{model_part_path.replace('/', '_')}{model_ckpt[:-5]}'''
    color = {0: 'white', 1: 'red', 2: 'green', 3: 'yellow'}

    scene_number = 0
    flight_number = 3
    for scene_number, traj in enumerate(labels):
        if scene_number in habitat.TRAIN:
            split = 'train'
        elif scene_number in habitat.VALID:
            split = 'val'
        else:
            split = 'test'
        for flight_number, label in traj.items():

            flight_folder = f'{habitat.get_flight_folder(scene_number, flight_number)}pred/'
            pred = np.load(f'{flight_folder}{name}.npz')
            loss = pred['loss']
            z_gt = pred['z_gt']
            z_hat = pred['z_hat']

            plt.plot(z_gt, label='z_gt')
            plt.plot(z_hat, label='z_hat')
            for i, e in enumerate(label):
                plt.axvspan(i - 0.5, i + 0.5, color=color[e])
            plt.grid()
            plt.ylabel('z [m]')
            plt.xlabel('Time [sample #]')
            plt.title(f'Scene_number {scene_number} Flight_number {flight_number} split {split}')
            plt.legend()
            plt.show()


def main():
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    for scene_number in habitat.TEST:
        for flight_number in labels[scene_number]:
            observe_traj_pred(scene_number, flight_number)
    # observe_generated_traj(24, 16)
    # boxplot_pred()
    # euroc_mav_table()
    # predict_good_fa()


if __name__ == '__main__':
    main()
