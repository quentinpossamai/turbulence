import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.transform import Transform as Tf
import src.data.habitat.loading as habitat
import src.data.euroc_mav.loading2 as euroc_mav
import src.data.habitat.data_gen as data_gen
import src.data.habitat.hatbitat2flu as hatbitat2flu
from src.drone_models.analytical_models.quadrotor_4parms import NumpyDynamicalModel
from src.utils.profiler import profiler


def plot_fa_law():
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
                                           past=h_prms['past'])

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
                                           past=h_prms['past'])

    elif name == 'emb':
        from src.drone_models.residual.emb_height import EmbHeight as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/AllHeight/20220223_121251/'
        model_name = 'epoch=136-step=13425-val_loss=0.50.ckpt'
        h_prms = pickle.load(open(f'{path}h_prms.pkl', 'rb'))
        model = Model.load_from_checkpoint(f'{path}{model_name}',
                                           activation=h_prms['activation'],
                                           embedding=h_prms['embedding'],
                                           l2=h_prms['l2'],
                                           learning_rate=h_prms['lr'],
                                           uav_model=uav_model,
                                           motor_size=h_prms['motor_size'],
                                           past=h_prms['past'])
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
                                           past=h_prms['past'])

    model.eval()

    z_mean, z_std = habitat.HabitatResidualDataset.z_mean, habitat.HabitatResidualDataset.z_std

    index = habitat.build_index_valid_flight(True)
    for scene_number in habitat.TEST:
        flight = index[scene_number]
        for flight_number in flight:
            state = torch.from_numpy(habitat.load_file('state', scene_number, flight_number).copy())
            fa = torch.from_numpy(habitat.load_file('ground_force', scene_number, flight_number).copy())
            z = torch.from_numpy(habitat.load_file('surface_dist', scene_number, flight_number).copy())
            motor = torch.from_numpy(habitat.load_file('motor', scene_number, flight_number).copy())
            depth_emb = torch.from_numpy(habitat.load_file('depth_emb', scene_number, flight_number).copy())
            depth = torch.from_numpy(habitat.load_file('depth', scene_number, flight_number).copy())

            assert len(fa) == len(motor)
            assert len(fa) + 1 == len(z)
            assert len(fa) + 1 == len(depth_emb)

            # fa = fa[h_prms['past'] - 1:-1, 2]
            xx = torch.arange(h_prms['past'] - 1, len(fa))
            fa_gt = fa[h_prms['past'] - 1:, 2]
            z_gt = z[h_prms['past'] - 1:-1, 0]
            assert len(xx) == len(fa_gt)
            assert len(fa_gt) == len(z_gt)

            with torch.no_grad():
                # fa_hat = torch.empty(len(fa), dtype=torch.float32) * torch.nan
                z_hat = torch.empty(len(z_gt), dtype=torch.float32) * torch.nan
                for i in range(len(z_hat)):
                    output = model(depth_emb[i:i + h_prms['past']])
                    z_hat[i] = (output * z_std + z_mean)
            print(f'{scene_number} | {flight_number} | '
                  f'Error loss wandb emb_height {((((z_hat - z) - z_mean) / z_std) ** 2).mean():0.4f} | '
                  f'Real loss {((z_hat - z) ** 2).mean():0.4f}')

            fig, ax = plt.subplots(1, 1)
            ax.set_title(f'Scene number: {scene_number} | flight_number: {flight_number}')
            ax.plot(xx, z_gt, label='Ground truth', marker=',')
            ax.plot(xx, z_hat, label='All height', marker=',')
            # ax.plot(fa_hat_a, label='model a')
            ax.grid()
            ax.set_ylabel('Surface distance (m)')  # Surface distance (m)
            ax.set_xlabel('Sample number')
            ax.legend()
            # plt.show()
            plt.savefig(f'{habitat.WD_PATH}reports/IROS_01_03_2022/graphics/model_prediction-{scene_number}-'
                        f'{flight_number}.pdf')


def rmse_model_prediction(name, dataset_name: str):
    msg = ['habitat', 'euroc_mav']
    assert dataset_name in msg, f'dataset must be in {msg}'
    use_cuda_if_possible = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_possible else 'cpu')

    msg = ['nogru', 'all', 'emb', 'fine']
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
    uav_model = NumpyDynamicalModel({'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=np.float32, g=9.81)

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
            return
        else:
            dataset = habitat.HabitatResidualDataset('test', h_prms['past'], False, True, False, False)
    else:
        assert name == 'fine'
        from src.drone_models.residual.fine_tuning2 import FineTunning as Model
        path = f'{habitat.WD_PATH}analysis/habitat_residual_training/FineTunning/20220224_183259/'
        model_name = 'epoch=0-step=49005-val_loss=0.49.ckpt'
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
                ground_force[2] = np.clip(np.sum(motor) ** 2 * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                          a_min=0, a_max=cap * drone_prms['m'] * g)
            return np.concatenate([ground_force, np.zeros(3)]).squeeze()[2]
    else:
        def compute_ground_effect(motor: np.ndarray, z: float) -> np.ndarray:
            sol = [9.40693193e-07, -4.17981623e-03]
            return sol[0] * motor.sum(axis=-1) / (z ** 2 - sol[1])

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
    print(name)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data), total=len(data)):
            if name == 'emb':
                depth = batch['depth_emb'].to(device)
            else:
                depth = batch['depth'].to(device)
            motor = batch['motor'].cpu().numpy()

            # z_hat = model(depth).cpu().numpy() * z_std + z_mean
            z_gt = batch['surface_dist'].cpu().numpy() * z_std + z_mean
            # loss = ((z_gt - z_hat) ** 2).squeeze()
            # z_loss[i] = loss
            # z_hat_log[i] = z_hat.squeeze()
            z_gt_log[i] = z_gt.squeeze()

            fa_gt = batch['ground_force'].cpu().numpy().squeeze()[2] * fa_std + fa_mean
            fa_gt_test = compute_ground_effect(motor, z_gt)
            # if dataset_name == 'habitat':
            #     assert np.isclose(fa_gt, fa_gt_test, atol=1e-8).all()
            # fa_hat = compute_ground_effect(motor, z_hat)
            # loss = (fa_gt - fa_hat) ** 2
            # fa_loss[i] = loss
            # fa_hat_log[i] = fa_hat
            fa_gt_log[i] = fa_gt
            fa_gttest_log[i] = fa_gt_test

    # pickle.dump(z_hat_log, open(f'{habitat.WD_PATH}z_hat_{name}_{dataset_name}.pkl', 'wb'))
    # pickle.dump(fa_hat_log, open(f'{habitat.WD_PATH}fa_hat_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(z_gt_log, open(f'{habitat.WD_PATH}z_gt_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(fa_gt_log, open(f'{habitat.WD_PATH}fag_gt_{name}_{dataset_name}.pkl', 'wb'))
    pickle.dump(fa_gttest_log, open(f'{habitat.WD_PATH}fag_gttest_{name}_{dataset_name}.pkl', 'wb'))

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(z_gt_log)
    # ax.plot(z_hat_log)
    # ax.legend()
    # plt.savefig(f'{habitat.WD_PATH}z_{name}_{dataset_name}.png')
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(fa_hat_log)
    # ax.plot(fa_gt_log)
    # ax.plot(fa_gttest_log)
    # ax.legend()
    # plt.savefig(f'{habitat.WD_PATH}fa_{name}_{dataset_name}.png')
    # print(f'rmse z_loss: {np.sqrt(z_loss.mean())}')
    # print(f'rmse fa_loss: {np.sqrt(fa_loss.mean())}')
    # print(f'\n\n')


def observe_generated_traj(scene_number, flight_number):
    scene_id = habitat.MP3D_SCENE_LIST[scene_number]
    scene_path = f"{habitat.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    cluster_scene_path = f"/storage/Datasets/3D-Simulator-Datasets/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    traj_path = f"{habitat.WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{habitat.FLIGHT}/{flight_number}/"
    state = habitat.load_file('state', scene_number, flight_number)
    state_habitat = hatbitat2flu.Habitat2FLU.state_flu_2_state_habitat(state)
    tf_follow = Tf.from_quat_trans(state_habitat[:, 3:7], state_habitat[:, :3]).as_matrix()

    command = f'rsync -tv remsto:{cluster_scene_path} {scene_path}'
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

    simulation = data_gen.Simulation(scene_path, camera_resolution=[480, 640], drone_prms=drone_prms, gravity=9.81)
    simulation.interactive_sim(camera_rgb=True, camera_depth=True, step=1 / 20, tf_follow=tf_follow)


def histogram():
    dataset = habitat.HabitatResidualDataset('test', 40, False, False, False, False)
    z_mean, z_std = dataset.z_mean, dataset.z_std
    fa_mean, fa_std = dataset.fa_mean, dataset.fa_std
    dataset_name = 'euroc_mav'  # habitat
    z_gt = pickle.load(open(f'{habitat.WD_PATH}z_gt_fine_{dataset_name}.pkl', 'rb'))
    fa_gt = pickle.load(open(f'{habitat.WD_PATH}fa_gt_fine_{dataset_name}.pkl', 'rb'))
    z_hat = pickle.load(open(f'{habitat.WD_PATH}z_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_hat = pickle.load(open(f'{habitat.WD_PATH}fa_hat_fine_{dataset_name}.pkl', 'rb'))




def main():
    # plot_fa_generator()
    # plot_fa()
    # ['nogru', 'all', 'emb', 'fine']
    # for e in ['nogru', 'all', 'emb', 'fine']:
    #     rmse_model_prediction(e, 'habitat')
    # for e in ['fine']:
    #     rmse_model_prediction(e, 'habitat')
    # observe_generated_traj(32, 3)
    histogram()


if __name__ == '__main__':
    main()
