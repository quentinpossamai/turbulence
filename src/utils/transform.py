import numpy as np
from scipy.spatial.transform import Rotation
from typing import Union


class Transform(object):
    def __init__(self):
        self.matrix: np.ndarray = np.eye(4)

    def __matmul__(self, multiple: Union['Transform', np.ndarray]) -> Union['Transform', np.ndarray]:
        if isinstance(multiple, Transform):
            return Transform().from_matrix(self.matrix @ multiple.get_matrix())
        else:  # type(multiple) is np.ndarray
            if multiple.shape[0] == 4:
                return self.matrix @ multiple
            elif multiple.shape in [(3,)]:
                return (self.matrix @ np.append(multiple, 1))[:-1]
            else:
                if not len(multiple.shape) in [1, 2]:
                    raise ArithmeticError(f"Multiplication between Transform object and object with "
                                          f"{len(multiple.shape)}-axis is not defined. Only with "
                                          f"shape =(4, n), (4,), (3,)")
                else:
                    raise ArithmeticError(f"Multiplication between Transform object and an object of shape "
                                          f"{multiple.shape} is not defined. Only with "
                                          f"shape =(4, n), (4,), (3,)")

    def __repr__(self):
        return self.matrix.__repr__()

    def get_rot_matrix(self) -> np.ndarray:
        return self.matrix[:3, :3]

    def get_rot_euler(self, seq, degrees):
        """
        Cf doc scipy.spatial.transform.Rotation
        """
        return Rotation.from_matrix(self.get_rot_matrix()).as_euler(seq=seq, degrees=degrees)

    def get_trans(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def get_quat(self) -> np.ndarray:
        """quat = x, y, z, w with scipy Rotation."""
        return Rotation.from_matrix(self.get_rot_matrix()).as_quat()

    def inv(self) -> 'Transform':
        r = self.get_rot_matrix()
        rt = r.T

        t = self.get_trans()
        t = - rt @ t

        new_tf = np.eye(4)
        new_tf[:3, :3] = rt
        new_tf[:3, 3] = t
        return Transform().from_matrix(new_tf)

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        A reference frame.

        B reference frame.

        A_R_B = quat = orientation of B relative to A.

        A_D_B = trans = translation from A origin to B origin described in A.

        P_A point described in A.

        A_T_B = tf = Transform object such as P_A = Transform().from_pose(trans, quat) @ P_B.

        trans, quat = tf.get_pose().

        :return: trans, quat.

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.                T_C_D = C_T_D
        """
        quat = Rotation.from_matrix(self.get_rot_matrix()).as_quat()
        trans = self.get_trans()
        return trans, quat

    def from_pose(self, trans: np.ndarray, quat: np.ndarray) -> 'Transform':
        """
        Create a Transform object from the reference frame of (xyz, quat) to a new one at position xyz and oriented
        with quat.

        quat = x, y, z, w

        A reference frame.

        B reference frame.

        A_R_B = quat = orientation of B relative to A.

        A_D_B = trans = translation from A origin to B origin described in A.

        P_A point described in A.

        A_T_B = tf = Transform().from_pose(trans, quat).

        P_A = tf @ P_B.

        :param trans: translation in 3d.
        :param quat: quaternions.
        :return: Transform object.

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        """
        assert trans.shape == (3,)
        assert quat.shape == (4,)
        assert np.round(np.linalg.norm(quat), 3) == 1

        r = Rotation.from_quat(quat).as_matrix()
        tf = np.eye(4)
        tf[:3, :3] = r
        tf[:3, 3] = trans
        self.matrix = tf
        return self

    def from_matrix(self, matrix: np.ndarray) -> 'Transform':
        self.matrix = matrix
        return self

    def from_euler(self, trans: np.ndarray, seq: str,
                   angles: np.ndarray, degrees: Optional[bool] = False) -> 'Transform':
        """
        Return a Transform object from a translation, an angle convention, and 3 Euler angles.

        :param trans: The translation of the Transform, must be of length 3 : A_D_B

        :param seq: Angle convention of the Euler angles, can be "xyz" or any other order.
        If "zyx" is used then angles must be = [z_angle, y_angle, x_angle]

        :param angles: An array of the 3 Euler angles in the order of the convention : A_R_B
        Meaning where the euler angles describes the position of B relative to A.

        :param degrees: True if the unit of the Euler angles is degree or False for radian

        :return: A_TF_B
        """
        assert angles.shape == (3,)
        return self.from_pose(trans=trans, quat=Rotation.from_euler(seq=seq, angles=angles, degrees=degrees).as_quat())

    def from_rot_matrix_trans_vect(self, trans: np.ndarray, rot: np.ndarray) -> 'Transform':
        """
        A reference frame.

        B reference frame.

        :param trans: B_D_A = translation from B origin to A origin described in B.

        :param rot: B_R_A = A 3x3 rotation matrix, orientation of A relative to B.

        P_A point described in A.

        :return: B_T_A = tf = Transform object such as
        P_B = Transform().from_rot_matrix_trans_vect(trans, quat) @ P_A.

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A, 1) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        Return the Transform object from a rotation matrix and a translation vector.
        """
        assert trans.shape == (3,)
        assert rot.shape == (3, 3)

        tf = np.eye(4)
        tf[:3, :3] = rot
        tf[:3, 3] = trans
        self.matrix = tf

        return self

    def from_trans_3_axis(self, trans: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> 'Transform':
        """

        :param x: unit vector of B x-axis expressed in A.
        :param y: unit vector of B y-axis expressed in A.
        :param z: unit vector of B z-axis expressed in A.
        :param trans: A_D_B = translation from A origin to B origin described in A.
        :return: The corresponding Transform object : A_tf_B
        """
        assert x.shape == (3,)
        assert y.shape == (3,)
        assert z.shape == (3,)
        assert trans.shape == (3,)

        # Inspired by https://stackoverflow.com/a/34392459/10054528
        a_r_b = np.hstack([x.reshape((3, 1)),  # r = A_R_B
                           y.reshape((3, 1)),
                           z.reshape((3, 1))])
        quat = Rotation.from_matrix(a_r_b).as_quat()
        self.from_pose(trans, quat)
        return self


def _test():
    pass


if __name__ == '__main__':
    _test()
