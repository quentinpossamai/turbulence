import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint
from tqdm import tqdm

from src.drone_models import drone_model
from src.utils.paths import WD_PATH


def test_drone_identification():
    # Dataset params https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_gazebo/resource/iris.yaml
    ground_truth_prms = {"m": torch.tensor(0.64),
                         "kt": torch.tensor(6.546e-6),
                         "km": torch.tensor(1.2864e-7),
                         "l_arm": torch.tensor(0.215),
                         "ixx": torch.tensor(10.007e-3),
                         "iyy": torch.tensor(10.2335e-3),
                         "izz": torch.tensor(8.1e-3)}

    print("Ground Truth params:")
    for key, val in ground_truth_prms.items():
        print(f"{key}: {val}")
    print()

    # tensorboard --logdir /Users/quentin/phd/turbulence/products/sanity check system identification/20210323_125158/
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = WD_PATH + f"products/sanity check system identification/{now}/"
    writer = SummaryWriter(result_folder)
    iteration = 0

    drone = drone_model.AscTecFireflyDroneModel(g=9.81, m=ground_truth_prms["m"],
                                                kt=ground_truth_prms["kt"],
                                                km=ground_truth_prms["km"],
                                                l_arm=ground_truth_prms["l_arm"],
                                                ixx=ground_truth_prms["ixx"],
                                                iyy=ground_truth_prms["iyy"],
                                                izz=ground_truth_prms["izz"])

    # Wrong initial params
    wrong_initial_prms = {"m": torch.tensor(0.64) / 2,
                          "kt": torch.tensor(6.546e-6) / 2,
                          "km": torch.tensor(1.2864e-7) / 2,
                          "l_arm": torch.tensor(0.215) / 2,
                          "ixx": torch.tensor(10.007e-3) / 2,
                          "iyy": torch.tensor(10.2335e-3) / 2,
                          "izz": torch.tensor(8.1e-3) / 2}
    print("Wrong initial params:")
    for key, val in wrong_initial_prms.items():
        print(f"{key}: {val}")
    print()

    def gen_cmd():
        w = torch.tensor([547.5904, 547.5904, 547.5904, 547.5904, 547.5904, 547.5904])
        w_randomized = w + (torch.rand(6) - 1) * w
        return lambda t: w_randomized

    dataset_nb = 10000

    time_array = torch.linspace(0, 2, 128)
    initial_state = torch.zeros(12)
    learned_prms = wrong_initial_prms
    for dataset_i in tqdm(range(dataset_nb)):
        # Generating the dataset
        w_func = gen_cmd()
        states = odeint(lambda t, x: drone.f(t_time=t, state=x, w_func=w_func, fa=torch.zeros(12)), initial_state,
                        time_array, method="euler")
        assert torch.isnan(states).sum() == 0
        # plot_simulated_drone_position(states, time_array, 0, 2)
        output = drone_model.system_identification(w_func, states, time_array, learned_prms,
                                                   batch_size=32,
                                                   epochs=1, lr=1e-3,
                                                   global_iteration=iteration,
                                                   writer=writer, verbose=False)
        learned_prms = {"m": output["m"][-1],
                        "kt": output["kt"][-1],
                        "km": output["km"][-1],
                        "l_arm": output["l_arm"][-1],
                        "ixx": output["ixx"][-1],
                        "iyy": output["iyy"][-1],
                        "izz": output["izz"][-1]}
        iteration = output["iteration"][-1]

    print("Learned params:")
    for key, val in learned_prms.items():
        print(f"{key}: {val}")
    print()
    pickle.dump(learned_prms, open(result_folder + f"learned_prms.pkl", "wb"))


def test_drone_reduced_identification():
    # Dataset params https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_gazebo/resource/iris.yaml
    original_prms = {"m": torch.tensor(1.56779),
                     "kt": torch.tensor(8.54858e-6),
                     "km": torch.tensor(1.6e-2),
                     "l_arm": torch.tensor(0.215),
                     "ixx": torch.tensor(0.0347563),
                     "iyy": torch.tensor(0.0458929),
                     "izz": torch.tensor(0.0977)}

    ground_truth_prms = {"theta1": original_prms["kt"] / original_prms["m"],
                         "theta2": original_prms["izz"] / original_prms["ixx"],
                         "theta3": original_prms["kt"] * original_prms["l_arm"] / original_prms["ixx"],
                         "theta4": original_prms["kt"] * original_prms["km"] / original_prms["izz"]}

    print("Ground Truth params:")
    for key, val in ground_truth_prms.items():
        print(f"{key}: {val}")
    print()

    # tensorboard --logdir "/Users/quentin/phd/turbulence/products/sanity check reduced system identification/20210326_173857/"
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = WD_PATH + f"products/sanity check reduced system identification/{now}/"
    writer = SummaryWriter(result_folder)
    iteration = 0

    ground_truth_drone = drone_model.AscTecFireflyReducedDroneModel(g=9.81, theta1=ground_truth_prms["theta1"],
                                                                    theta2=ground_truth_prms["theta2"],
                                                                    theta3=ground_truth_prms["theta3"],
                                                                    theta4=ground_truth_prms["theta4"])

    # Wrong initial params
    wrong_initial_prms = {"theta1": ground_truth_prms["theta1"] / 2,
                          "theta2": ground_truth_prms["theta2"] / 2,
                          "theta3": ground_truth_prms["theta3"] / 2,
                          "theta4": ground_truth_prms["theta4"] / 2}
    print("Wrong initial params:")
    for key, val in wrong_initial_prms.items():
        print(f"{key}: {val}")
    print()

    def gen_cmd():
        w = torch.tensor([547.5904, 547.5904, 547.5904, 547.5904, 547.5904, 547.5904])
        w_randomized = w + (torch.rand(6) - 1) * w
        return lambda t: w_randomized

    dataset_nb = 10000

    time_array = torch.linspace(0, 2, 128)
    initial_state = torch.zeros(12)
    learned_prms = wrong_initial_prms
    for dataset_i in tqdm(range(dataset_nb)):
        # Generating the dataset
        w_func = gen_cmd()
        states = odeint(lambda t, x: ground_truth_drone.f(t_time=t, state=x, w_func=w_func), initial_state,
                        time_array, method="euler")
        assert torch.isnan(states).sum() == 0
        # plot_simulated_drone_position(states, time_array, 0, 2)
        output = drone_model.system_identification_reduced_model(w_func, states, time_array, learned_prms,
                                                                 batch_size=32,
                                                                 epochs=1, lr=1e-3,
                                                                 global_iteration=iteration,
                                                                 writer=writer, verbose=False)

        learned_prms = {"theta1": output["theta1"][-1],
                        "theta2": output["theta2"][-1],
                        "theta3": output["theta3"][-1],
                        "theta4": output["theta4"][-1]}

        iteration = output["iteration"][-1]

    print("Learned params:")
    for key, val in learned_prms.items():
        print(f"{key}: {val}")
    print()
    pickle.dump(learned_prms, open(result_folder + f"learned_prms.pkl", "wb"))


def analyze_test_result():
    result_folder = WD_PATH + f"products/sanity check system identification/"
    last_folder = sorted(next(os.walk(result_folder))[1])[-1] + "/"
    prm = pickle.load(open(result_folder + last_folder + "learned_prms.pkl", "rb"))

    ground_truth_prms = {"m": torch.tensor(1.56779),
                         "kt": torch.tensor(8.54858e-6),
                         "km": torch.tensor(1.6e-2),
                         "l_arm": torch.tensor(0.215),
                         "ixx": torch.tensor(0.0347563),
                         "iyy": torch.tensor(0.0458929),
                         "izz": torch.tensor(0.0977)}

    learned_prms = {"m": 0.41721704602241516,
                    "kt": 4.266622454451863e-06,
                    "km": 6.433720045606606e-08,
                    "l_arm": 0.14422546327114105,
                    "ixx": 0.00437798211351037,
                    "iyy": 0.004471251741051674,
                    "izz": 0.003541051410138607}

    drone_truth = drone_model.AscTecFireflyDroneModel(g=9.81, m=ground_truth_prms["m"],
                                                      kt=ground_truth_prms["kt"],
                                                      km=ground_truth_prms["km"],
                                                      l_arm=ground_truth_prms["l_arm"],
                                                      ixx=ground_truth_prms["ixx"],
                                                      iyy=ground_truth_prms["iyy"],
                                                      izz=ground_truth_prms["izz"])
    drone_learned = drone_model.AscTecFireflyDroneModel(g=9.81, m=learned_prms["m"],
                                                        kt=learned_prms["kt"],
                                                        km=learned_prms["km"],
                                                        l_arm=learned_prms["l_arm"],
                                                        ixx=learned_prms["ixx"],
                                                        iyy=learned_prms["iyy"],
                                                        izz=learned_prms["izz"])
    time_array = torch.linspace(0, 2, 128)
    initial_state = torch.zeros(12)

    def gen_cmd():
        w = torch.tensor([547.5904, 547.5904, 547.5904, 547.5904, 547.5904, 547.5904])
        w_randomized = w + (torch.rand(6) - 1) * w
        return lambda t: w_randomized

    w_func = gen_cmd()
    states_truth = odeint(lambda t, x: drone_truth.f(t_time=t, state=x, w_func=w_func, fa=torch.zeros(12)),
                          initial_state, time_array, method="euler")
    states_learned = odeint(lambda t, x: drone_learned.f(t_time=t, state=x, w_func=w_func, fa=torch.zeros(12)),
                            initial_state, time_array, method="euler")

    # for name, func in [("orientation", plot_simulated_drone_orientation), ("position", plot_simulated_drone_position),
    #                    ("lin_spd", plot_simulated_drone_lin_spd), ("ang_vel", plot_simulated_drone_ang_vel)]:
    for name, func in [("position", plot_simulated_drone_position), ("lin_spd", plot_simulated_drone_lin_spd)]:
        start, end = 0, 1
        fig_axs = func(states_truth, time_array, start, end, plot=False)
        func(states_learned, time_array, start, end, saving_path_name=result_folder + last_folder + name + ".png",
             fig_axs=fig_axs, plot=True)
    plot_motor_speed(time_array, [w_func(0) for e in time_array],
                     saving_path_name=result_folder + last_folder + "motor_speed" + ".png")

    print()


def analyze_test_reduced_result():
    result_folder = WD_PATH + f"products/sanity check reduced system identification/"
    last_folder = sorted(next(os.walk(result_folder))[1])[-1] + "/"
    prm = pickle.load(open(result_folder + last_folder + "learned_prms.pkl", "rb"))

    original_prms = {"m": torch.tensor(1.56779),
                     "kt": torch.tensor(8.54858e-6),
                     "km": torch.tensor(1.6e-2),
                     "l_arm": torch.tensor(0.215),
                     "ixx": torch.tensor(0.0347563),
                     "iyy": torch.tensor(0.0458929),
                     "izz": torch.tensor(0.0977)}

    ground_truth_prms = {"theta1": original_prms["kt"] / original_prms["m"],
                         "theta2": original_prms["izz"] / original_prms["ixx"],
                         "theta3": original_prms["kt"] * original_prms["l_arm"] / original_prms["ixx"],
                         "theta4": original_prms["kt"] * original_prms["km"] / original_prms["izz"]}

    learned_prms = {"theta1": 0,
                    "theta2": np.nan,
                    "theta3": 0,
                    "theta4": 0}

    drone_truth = drone_model.AscTecFireflyReducedDroneModel(g=9.81, theta1=ground_truth_prms["theta1"],
                                                             theta2=ground_truth_prms["theta2"],
                                                             theta3=ground_truth_prms["theta3"],
                                                             theta4=ground_truth_prms["theta4"])
    drone_learned = drone_model.AscTecFireflyReducedDroneModel(g=9.81, theta1=learned_prms["theta1"],
                                                               theta2=learned_prms["theta2"],
                                                               theta3=learned_prms["theta3"],
                                                               theta4=learned_prms["theta4"])
    time_array = torch.linspace(0, 2, 128)
    initial_state = torch.zeros(12)

    def gen_cmd():
        w = torch.tensor([547.5904, 547.5904, 547.5904, 547.5904, 547.5904, 547.5904])
        w_randomized = w + (torch.rand(6) - 1) * w
        return lambda t: w_randomized

    w_func = gen_cmd()
    states_truth = odeint(lambda t, x: drone_truth.f(t_time=t, state=x, w_func=w_func),
                          initial_state, time_array, method="euler")
    states_learned = odeint(lambda t, x: drone_learned.f(t_time=t, state=x, w_func=w_func),
                            initial_state, time_array, method="euler")

    for name, func in [("orientation", plot_simulated_drone_orientation), ("position", plot_simulated_drone_position),
                       ("lin_spd", plot_simulated_drone_lin_spd), ("ang_vel", plot_simulated_drone_ang_vel)]:
        # for name, func in [ ("position", plot_simulated_drone_position), ("lin_spd", plot_simulated_drone_lin_spd)]:
        start, end = 0, 2
        fig_axs = func(states_truth, time_array, start, end, plot=False)
        func(states_learned, time_array, start, end, saving_path_name=result_folder + last_folder + name + ".png",
             fig_axs=fig_axs, plot=True)
    plot_motor_speed(time_array, [w_func(0) for e in time_array],
                     saving_path_name=result_folder + last_folder + "motor_speed" + ".png")

    print()


def plot_simulated_drone_orientation(states, t_tab, t_start, t_end, saving_path_name: str = None, fig_axs=None,
                                     plot=True):
    if fig_axs is None:
        fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    else:
        fig, axs = fig_axs
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
    if plot:
        plt.show()
    return fig, axs


def plot_simulated_drone_position(states, t_tab, t_start, t_end, saving_path_name: str = None, fig_axs=None,
                                  plot=True):
    if fig_axs is None:
        fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    else:
        fig, axs = fig_axs
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
    if plot:
        plt.show()
    return fig, axs


def plot_simulated_drone_lin_spd(states, t_tab, t_start, t_end, saving_path_name: str = None, fig_axs=None,
                                 plot=True):
    if fig_axs is None:
        fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    else:
        fig, axs = fig_axs
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
    if plot:
        plt.show()
    return fig, axs


def plot_simulated_drone_ang_vel(states, t_tab, t_start, t_end, saving_path_name: str = None, fig_axs=None,
                                 plot=True):
    if fig_axs is None:
        fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
    else:
        fig, axs = fig_axs
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
    if plot:
        plt.show()
    return fig, axs


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
    # test_drone_identification()
    test_drone_reduced_identification()
    # analyze_test_result()
