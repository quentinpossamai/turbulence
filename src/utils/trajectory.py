import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from scipy.spatial.transform import Rotation


#     plt.rcParams.update({"text.usetex": True,
#                          "font.family": "serif",
#                          "font.sans-serif": ["Computer Modern Roman"]})

def trace_trajectory3d(position: np.ndarray, quaternion: np.ndarray, pause: float, step: int,
                       position_ref: np.ndarray = None):
    # transforms = Transform.from_quat_trans(quaternion, position).inv().inv().as_matrix()
    # position = Transform.from_matrix(transforms).as_trans()
    # quaternion = Transform.from_matrix(transforms).as_rot_quat()
    state = np.concatenate([position, quaternion], axis=-1)
    print(len(state))

    def get_body_frame_point(pos, quat):
        width = 1
        d1 = np.array([1, 0, 0])
        d2 = np.array([0, 1, 0])
        d3 = np.array([0, 0, 1])

        rot = Rotation.from_quat(quat).as_matrix()
        return pos + rot @ d1 * width, pos + rot @ d2 * width, pos + rot @ d3 * width

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()
    for t in range(0, state.shape[0], step):
        ax.clear()
        ax.plot(state[:, 0], state[:, 1], state[:, 2])
        if position_ref is not None:
            ax.plot(position_ref[:, 0], position_ref[:, 1], position_ref[:, 2])
        ax.set_xlim(-3 + state[t, 0], state[t, 0] + 3)
        ax.set_ylim(-3 + state[t, 1], state[t, 1] + 3)
        ax.set_zlim(-3 + state[t, 2], state[t, 2] + 3)
        p1, p2, p3 = get_body_frame_point(state[t, :3], state[t, 3:])

        ax.plot([state[t, 0], p1[0]], [state[t, 1], p1[1]], [state[t, 2], p1[2]], c='red')
        ax.plot([state[t, 0], p2[0]], [state[t, 1], p2[1]], [state[t, 2], p2[2]], c='green')
        ax.plot([state[t, 0], p3[0]], [state[t, 1], p3[1]], [state[t, 2], p3[2]], c='blue')
        ax.set_title(str(t))
        plt.pause(pause)
    plt.show()


def timestamp_analysis(timestamps):
    dt = timestamps[1:] - timestamps[:-1]
    mean = np.mean(dt)
    std = np.std(dt)

    fig, axs = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=100)
    ax = axs[0]
    ax.plot(timestamps[1:], dt)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Time steps [s]")
    ax.grid()

    ax = axs[1]
    ax.hist(dt, bins=1000)
    ax.set_title(f"Mean and std: {mean:.4E} $\pm$ {std:.4E}")
    ax.set_xlabel("Time steps [s]")
    ax.set_ylabel("# of sample")
    ax.grid()

    plt.tight_layout()
    plt.show()


def compare_traj(state_gt, state_hat, time, start, stop, traj_name, save_path=None):
    """Plotting trajectory with model vs ground truth"""
    plt.rcParams.update({"text.usetex": True,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})
    if save_path is not None:
        fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    else:
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

    # Position
    for i, name in zip(range(3), ["x", "y", "z"]):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.plot(time[start:stop], state_gt[start:stop, i], label=f"GT {name}")
        ax.plot(time[start:stop], state_hat[start:stop, i], label=f"hat {name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.grid()
        ax.legend()
        if i == 1:
            ax.set_title(traj_name)

    # Linear speed
    for i, name in zip(range(7, 10), ["x", "y", "z"]):
        ax = fig.add_subplot(4, 3, i)
        ax.plot(time[start:stop], state_gt[start:stop, i], label=f"GT {name}")
        ax.plot(time[start:stop], state_hat[start:stop, i], label=f"hat {name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Linear speed [m/s]")
        ax.grid()
        ax.legend()

    # Angular velocity
    for i, name in zip(range(10, 13), ["x", "y", "z"]):
        ax = fig.add_subplot(4, 3, i)
        ax.plot(time[start:stop], state_gt[start:stop, i], label=f"GT {name}")
        ax.plot(time[start:stop], state_hat[start:stop, i], label=f"hat {name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular velocity [rad/s]")
        ax.grid()
        ax.legend()
    plt.tight_layout()

    # Quaternion
    for i, name in zip(range(3, 7), ["q0", "q1", "q2", "q3"]):
        ax = fig.add_subplot(4, 4, i + 2)
        ax.plot(time[start:stop], state_gt[start:stop, i], label=f"GT {name}")
        ax.plot(time[start:stop], state_hat[start:stop, i], label=f"hat {name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Quaternion")
        ax.grid()
        ax.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": ['sans-serif'],
                         "font.sans-serif": ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif',
                                             'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica',
                                             'Avant Garde', 'sans-serif']})


def gen_traj(model, state, motor, dt, horizon):
    assert isinstance(horizon, int) and horizon > 0
    if horizon == 1:
        with torch.no_grad():
            next_state_hat = model(state, motor, dt.reshape((-1, 1)))
        return next_state_hat
    elif horizon > 1:
        # With integration
        traj = [state[0]]
        with torch.no_grad():
            for i in range(len(state) - 1):
                traj.append(model(traj[-1].unsqueeze(0), motor[i, :].unsqueeze(0), dt[i].reshape((1, 1))).squeeze())
        traj_stack = torch.stack(traj, dim=0)
        return traj_stack
