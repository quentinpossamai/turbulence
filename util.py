from __future__ import print_function
import sys
import numpy as np
from typing import Union, Tuple
from scipy.spatial.transform import Rotation


class Progress(object):
    def __init__(self, max_iter, end_print=''):
        self.max_iter = max_iter
        self.end_print = end_print
        self.iter = 0

    def update_pgr(self, iteration=None):
        if iteration is not None:
            self.iter = iteration
        # Progression
        print('\rProgression : {:0.02f}%'.format((self.iter + 1) * 100 / self.max_iter), end='')
        sys.stdout.flush()
        if self.iter + 1 == self.max_iter:
            print(self.end_print)
        self.iter += 1


class Transform(object):
    def __init__(self):
        self.matrix: np.ndarray = np.eye(4)

    def __matmul__(self, multiple: Union['Transform', np.ndarray]) -> Union['Transform', np.ndarray]:
        if type(multiple) is Transform:
            return Transform().from_matrix(self.matrix @ multiple.get_matrix())
        else:
            return self.matrix @ multiple

    def get_rot(self) -> np.ndarray:
        return self.matrix[:3, :3]

    def get_trans(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def get_inv(self) -> 'Transform':
        r = self.get_rot()
        rt = np.transpose(r)

        t = self.get_trans()
        t = - rt @ t

        new_tf = np.eye(4)
        new_tf[:3, :3] = rt
        new_tf[:3, 3] = t
        return Transform().from_matrix(new_tf)

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        quat = Rotation.from_matrix(self.get_rot()).as_quat()
        xyz = self.get_trans()
        return xyz, quat

    def from_pos(self, xyz: np.ndarray, quat: np.ndarray) -> 'Transform':
        """
        Create a Transform object from the reference frame of (xyz, quat) to a new one at position xyz and oriented with
        quat
        :param xyz: position in 3d
        :param quat: quaternions
        :return: Transform
        """
        assert xyz.shape == (3,)
        assert quat.shape == (4,)
        if all(quat == np.zeros(4)):
            r = np.eye(3)
        else:
            r = Rotation.from_quat(quat).as_matrix()
        r = np.transpose(r)
        t = - r @ xyz
        tf = np.eye(4)
        tf[:3, :3] = r
        tf[:3, 3] = t
        self.matrix = tf
        return self

    def from_matrix(self, matrix: np.ndarray) -> 'Transform':
        self.matrix = matrix
        return self


def is_list_array_unique(list_arrays):
    """
    Compare if two list of numpy arrays are equal element-wise.
    :param list_arrays: List of numpy array
    :return: boolean
    """
    a = list_arrays[0]
    return all(np.array_equal(a, x) for x in list_arrays)


def angle_arccos(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    return np.arccos((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))