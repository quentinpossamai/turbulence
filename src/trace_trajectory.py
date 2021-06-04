import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pickle
from scipy.spatial.transform import Rotation

from src.utils.transform import Transform
from src.utils.paths import WD_PATH


def trace_trajectory(transforms: np.ndarray):
    def get_vertex_point(pos, quat):
        width = 0.5
        d1 = np.array([1, 0, 0])
        d2 = np.array([0, 1, 0])
        d3 = np.array([0, 0, 1])

        r = Rotation.from_quat(quat)
        d1, d2, d3 = r.apply(d1), r.apply(d2), r.apply(d3)

        p1 = pos + d1 * width
        p2 = pos + d2 * width
        p3 = pos + d3 * width

        return p1, p2, p3

    position = Transform.from_matrix(transforms).as_trans()
    quaternion = Transform.from_matrix(transforms).as_rot_quat()
    norm = np.sqrt(np.sum(quaternion ** 2, axis=-1, keepdims=True))
    plt.plot(norm)
    plt.ylim(0, 1.2)
    plt.show()

    quaternion = quaternion  # / np.sqrt(np.sum(quaternion ** 2, axis=-1, keepdims=True))
    state = np.concatenate([position, quaternion], axis=-1)[10000:]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()
    for t in range(0, state.shape[0], 2):
        ax.clear()
        ax.plot(state[:, 0], state[:, 1], state[:, 2])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(0, 3)
        p1, p2, p3 = get_vertex_point(state[t, :3], state[t, 3:])

        ax.plot([state[t, 0], p1[0]], [state[t, 1], p1[1]], [state[t, 2], p1[2]], c='red')
        ax.plot([state[t, 0], p2[0]], [state[t, 1], p2[1]], [state[t, 2], p2[2]], c='blue')
        ax.plot([state[t, 0], p3[0]], [state[t, 1], p3[1]], [state[t, 2], p3[2]], c='green')
        ax.set_title(str(t))
        plt.pause(0.000001)
    plt.show()


if __name__ == '__main__':
    trace_trajectory()
