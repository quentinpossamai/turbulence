import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.transform import Transform


def trace_trajectory(position, quaternion: np.ndarray):
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
    for t in range(6000, state.shape[0], 20):
        ax.clear()
        ax.plot(state[:, 0], state[:, 1], state[:, 2])
        ax.set_xlim(-3 + state[t, 0], state[t, 0] + 3)
        ax.set_ylim(-3 + state[t, 1], state[t, 1] + 3)
        ax.set_zlim(-3 + state[t, 2], state[t, 2] + 3)
        p1, p2, p3 = get_body_frame_point(state[t, :3], state[t, 3:])

        ax.plot([state[t, 0], p1[0]], [state[t, 1], p1[1]], [state[t, 2], p1[2]], c='red')
        ax.plot([state[t, 0], p2[0]], [state[t, 1], p2[1]], [state[t, 2], p2[2]], c='green')
        ax.plot([state[t, 0], p3[0]], [state[t, 1], p3[1]], [state[t, 2], p3[2]], c='blue')
        ax.set_title(str(t))
        plt.pause(0.0001)
    plt.show()
