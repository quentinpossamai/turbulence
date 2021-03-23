import torch
from torchdiffeq import odeint

import code_python.drone_model as drone_model


def test_drone_identification():
    m = torch.tensor(0.64)  # Papier Asctec
    kt = torch.tensor(6.546e-6)  # Papier 2
    km = torch.tensor(1.2864e-7)  # Papier 2
    l_arm = torch.tensor(0.215)  # Papier 2
    ixx = torch.tensor(10.007e-3)  # Papier 2
    iyy = torch.tensor(10.2335e-3)  # Papier 2
    izz = torch.tensor(8.1e-3)  # Papier 2

    drone = drone_model.AscTecFireflyDroneModel(g=9.81, m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy, izz=izz)

    def gen_cmd():
        w = torch.tensor([547.5904, 547.5904, 547.5904, 547.5904, 547.5904, 547.5904])
        w_randomized = w + (torch.rand(6) - 1) * w
        return lambda t: w_randomized

    w_func = gen_cmd()
    dataset_nb = 10

    time_array = torch.linspace(0, 5, 100)
    initial_state = torch.zeros(12)
    for dataset_i in range(dataset_nb):
        states = odeint(lambda t, x: drone.f(t_time=t, state=x, w_func=w_func, fa=torch.zeros(12)), initial_state,
                        time_array, method="euler")

    to_plot = drone_model.estimate_euroc_mav_params_h1(flight_number=0, epochs=1, batch_size=32, lr=1e-2,
                                                       load_previous=False, dataset=[states, w])
    return 1


if __name__ == "__main__":
    test_drone_identification()
