import numpy as np
import torch

from src.drone_models import quadrotor


def system_identification(u_func: torch.tensor, g_states: torch.tensor, time_train: torch.tensor,
                          initial_parameters, batch_size, epochs, lr, global_iteration, writer=None,
                          verbose=True):
    """
    Learn drone's parameters
    :param u_func: Function that generate a command of the system, time as a parameter.
    :param g_states: Ground truth states.
    :param time_train: Time of the states.
    :param initial_parameters: The initial parameters used to start the optimization.
    :param batch_size: Number of data to mean over before a backpropagation.
    :param epochs: Number of time data are optimized over.
    :param lr: Learning rate used for Adam optimizer.
    :param global_iteration: Iteration to use for plotting data on tensorboard.
    :param writer: tensorboard SummaryWriter
    :param verbose: True if you want console printed info.
    :return: Drone's parameters identified to data.
    """
    m, kt, km, l_arm, ixx, iyy, izz = initial_parameters.values()

    # Parameter's scaling coefficients initialization
    prm_name = ["m", "kt", "km", "l_arm", "ixx", "iyy", "izz"]  # Parameters name

    coef_m = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_kt = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_km = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_l_arm = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_ixx = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_iyy = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_izz = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))

    saved_vars = {"loss": [], "iteration": [global_iteration]}
    for e in prm_name:
        exec(f"""saved_vars["{e}"] = [(coef_{e} * {e}).item()]""")
        if writer is not None:
            exec(f"""writer.add_scalars("{e}", {{"{e}": (coef_{e} * {e}).item()}}, global_step=global_iteration)""")
    drone_model = AscTecFireflyDroneModel(coef_m=coef_m, g=9.81, coef_kt=coef_kt, coef_km=coef_km,
                                          coef_l_arm=coef_l_arm, coef_ixx=coef_ixx, coef_iyy=coef_iyy,
                                          coef_izz=coef_izz, m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy,
                                          izz=izz)

    if verbose:
        for e in prm_name:
            exec(f"""print(f"scaled: {{drone_model.coef_{e}}} | val: {{drone_model.{e}}} |"""
                 f"""new val: {{drone_model.coef_{e}*drone_model.{e}}}")""")

    # TODO use dataloader or shuffle data

    # Adam optimization
    optimizer = torch.optim.Adam(drone_model.parameters(), lr=lr)
    batch_nb = len(time_train) // batch_size
    for epoch in range(epochs):
        for batch_i in (tqdm(range(batch_nb + 1),
                             desc=f"Epoch nº{epoch + 1}/{epochs}") if verbose else range(batch_nb + 1)):
            # Dividing data by batch
            start = batch_i * batch_size - 1
            end = (batch_i + 1) * batch_size
            if batch_i == batch_nb:
                time_batch = time_train[start:]
                g_states_batch = g_states[start:]
            elif batch_i == 0:
                time_batch = time_train[start + 1:end]
                g_states_batch = g_states[start + 1:end]
            else:
                time_batch = time_train[start:end]
                g_states_batch = g_states[start:end]

            if len(time_batch) == 1:  # If batch is size 1 then pass.
                continue

            batch_loss = torch.zeros(1)
            # Forward
            for tn, tn1, g_n, g_n1 in zip(time_batch[:-1], time_batch[1:], g_states_batch[:-1], g_states_batch[1:]):
                dt = tn1 - tn
                state_n1 = drone_model.f(tn, state=g_n, w_func=u_func, fa=torch.zeros(12)) * dt + g_n
                loss = (g_n1 - state_n1) ** 2
                coef_learning = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
                loss = (coef_learning / coef_learning.sum() * loss).sum()
                batch_loss = loss + batch_loss

            batch_loss = batch_loss / (len(g_states_batch) - 1)

            optimizer.zero_grad()  # zero the gradient buffers
            batch_loss.backward()  # And learning rate
            optimizer.step()  # Does the update

            # Saving data
            if writer is not None:
                writer.add_scalars("Loss", {"Training loss": batch_loss.item()}, global_step=global_iteration)
            saved_vars["loss"].append(batch_loss.item())
            for e in prm_name:
                exec(f"""saved_vars["{e}"].append((coef_{e} * {e}).item())""")
                if writer is not None:
                    exec(f"""writer.add_scalars("{e}", {{"{e}": (coef_{e} * {e}).item()}},
                             global_step=global_iteration)""")
            global_iteration += 1
            saved_vars["iteration"].append(global_iteration)

    if verbose:
        for e in prm_name:
            exec(f"""print(f"scaled: {{drone_model.coef_{e}}} | val: {{drone_model.{e}}} |"""
                 f"""new val: {{drone_model.coef_{e}*drone_model.{e}}}")""")

    return saved_vars


if __name__ == "__main__":
    system_identification()
