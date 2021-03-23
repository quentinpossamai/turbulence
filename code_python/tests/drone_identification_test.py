import torch
from torchdiffeq import odeint
import datetime
import pickle
import matplotlib.pyplot as plt

import code_python.drone_model as drone_model
from code_python.utils import TURBULENCE_PATH


def test_drone_identification():
    # Dataset params https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_gazebo/resource/iris.yaml
    m = torch.tensor(0.64)
    kt = torch.tensor(6.546e-6)
    km = torch.tensor(1.2864e-7)
    l_arm = torch.tensor(0.215)
    ixx = torch.tensor(10.007e-3)
    iyy = torch.tensor(10.2335e-3)
    izz = torch.tensor(8.1e-3)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = TURBULENCE_PATH + f"products/sanity check system identification/{now}/"

    drone = drone_model.AscTecFireflyDroneModel(g=9.81, m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy, izz=izz)

    # Wrong initial params
    m = torch.tensor(0.64) / 2
    kt = torch.tensor(6.546e-6) / 2
    km = torch.tensor(1.2864e-7) / 2
    l_arm = torch.tensor(0.215) / 2
    ixx = torch.tensor(10.007e-3) / 2
    iyy = torch.tensor(10.2335e-3) / 2
    izz = torch.tensor(8.1e-3) / 2
    initial_parameters = (m, kt, km, l_arm, ixx, iyy, izz)

    def gen_cmd():
        w = torch.tensor([547.5904, 547.5904, 547.5904, 547.5904, 547.5904, 547.5904])
        w_randomized = w + (torch.rand(6) - 1) * w
        return lambda t: w_randomized

    dataset_nb = 1000

    time_array = torch.linspace(0, 2, 128)
    initial_state = torch.zeros(12)
    for dataset_i in range(dataset_nb):
        # Generating the dataset
        w_func = gen_cmd()
        states = odeint(lambda t, x: drone.f(t_time=t, state=x, w_func=w_func, fa=torch.zeros(12)), initial_state,
                        time_array, method="euler")
        assert torch.isnan(states).sum() == 0
        # plot_simulated_drone_position(states, time_array, 0, 2)
        learned_prms = drone_model.system_identification(w_func, states, time_array, initial_parameters, batch_size=32,
                                                         epochs=1, lr=1e-3,
                                                         global_iteration=dataset_i * len(time_array),
                                                         tensorboard_path=result_folder, verbose=False)
        initial_parameters = (learned_prms["m"][-1],
                              learned_prms["kt"][-1],
                              learned_prms["km"][-1],
                              learned_prms["l_arm"][-1],
                              learned_prms["ixx"][-1],
                              learned_prms["iyy"][-1],
                              learned_prms["izz"][-1])

    pickle.dump(learned_prms, open(result_folder + f"learned_prms.pkl", "wb"))

    return 1


def plot_simulated_drone_orientation(states, t_tab, t_start, t_end, saving_path_name: str = None):
    fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    inds = [i for i, e in enumerate(t_tab) if t_start < e < t_end]
    axs[0].plot(t_tab[inds], states[:, 0][inds].detach(), c='r')
    axs[0].set_ylabel("phi(rad)")
    axs[1].plot(t_tab[inds], states[:, 1][inds].detach(), c='b')
    axs[1].set_ylabel("theta(rad)")
    axs[2].plot(t_tab[inds], states[:, 2][inds].detach(), c='g')
    axs[2].set_ylabel("psi(rad)")
    for ax in axs:
        ax.grid()
        ax.set_xlabel("Time(s)")
    plt.tight_layout()
    if saving_path_name is not None:
        plt.savefig(saving_path_name)
    plt.show()


def plot_simulated_drone_position(states, t_tab, t_start, t_end, saving_path_name: str = None):
    fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    inds = [i for i, e in enumerate(t_tab) if t_start < e < t_end]
    axs[0].plot(t_tab[inds], states[:, 9][inds].detach(), c='r')
    axs[0].set_ylabel("x(m)")
    axs[1].plot(t_tab[inds], states[:, 10][inds].detach(), c='b')
    axs[1].set_ylabel("y(m)")
    axs[2].plot(t_tab[inds], states[:, 11][inds].detach(), c='g')
    axs[2].set_ylabel("z(m)")
    for ax in axs:
        ax.grid()
        ax.set_xlabel("Time(s)")
    plt.tight_layout()
    if saving_path_name is not None:
        plt.savefig(saving_path_name)
    plt.show()


def plot_simulated_drone_lin_spd(states, t_tab, t_start, t_end, saving_path_name: str = None):
    fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    inds = [i for i, e in enumerate(t_tab) if t_start < e < t_end]
    axs[0].plot(t_tab[inds], states[:, 6][inds].detach(), c='r')
    axs[0].set_ylabel("u(m/s)")
    axs[1].plot(t_tab[inds], states[:, 7][inds].detach(), c='b')
    axs[1].set_ylabel("v(m/s)")
    axs[2].plot(t_tab[inds], states[:, 8][inds].detach(), c='g')
    axs[2].set_ylabel("w(m/s)")
    for ax in axs:
        ax.grid()
        ax.set_xlabel("Time(s)")
    plt.tight_layout()
    if saving_path_name is not None:
        plt.savefig(saving_path_name)
    plt.show()


def plot_simulated_drone_ang_vel(states, t_tab, t_start, t_end, saving_path_name: str = None):
    fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    inds = [i for i, e in enumerate(t_tab) if t_start < e < t_end]
    axs[0].plot(t_tab[inds], states[:, 3][inds].detach(), c='r')
    axs[0].set_ylabel("p(rad/s)")
    axs[1].plot(t_tab[inds], states[:, 4][inds].detach(), c='b')
    axs[1].set_ylabel("q(rad/s)")
    axs[2].plot(t_tab[inds], states[:, 5][inds].detach(), c='g')
    axs[2].set_ylabel("r(rad/s)")
    for ax in axs:
        ax.grid()
        ax.set_xlabel("Time(s)")
    plt.tight_layout()
    if saving_path_name is not None:
        plt.savefig(saving_path_name)
    plt.show()


def plot_motor_speed(t_tab, motor_speed, saving_path_name: str = None):
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    for i in range(6):
        ax.plot(t_tab, [e[i] for e in motor_speed], label=f"{i}")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if saving_path_name is not None:
        plt.savefig(saving_path_name)
    plt.show()


if __name__ == "__main__":
    test_drone_identification()
