import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple


class Transform(object):
    def __init__(self, matrix: np.ndarray):
        self._matrix: np.ndarray = matrix  # shape = (n, 4, 4)
        self._is_single = False
        if self._matrix.shape[0] == 1:
            self._is_single = True

    # def __matmul__(self, multiple: Union['Transform', np.ndarray]) -> Union['Transform', np.ndarray]:
    #     if isinstance(multiple, Transform):
    #         return Transform().from_matrix(self.matrix @ multiple.as_matrix())
    #     else:  # type(multiple) is np.ndarray
    #         if multiple.shape[0] == 4:
    #             return self.matrix @ multiple
    #         elif multiple.shape in [(3,)]:
    #             return (self.matrix @ np.append(multiple, 1))[:-1]
    #         else:
    #             if not len(multiple.shape) in [1, 2]:
    #                 raise ArithmeticError(f"Multiplication between Transform object and object with "
    #                                       f"{len(multiple.shape)}-axis is not defined. Only with "
    #                                       f"shape =(4, n), (4,), (3,)")
    #             else:
    #                 raise ArithmeticError(f"Multiplication between Transform object and an object of shape "
    #                                       f"{multiple.shape} is not defined. Only with "
    #                                       f"shape =(4, n), (4,), (3,)")
    #
    def __repr__(self):
        return self._matrix.__repr__()

    def __len__(self):
        """Number of rotations contained in this object.

        Multiple rotations can be stored in a single instance.

        Returns
        -------
        length : int
            Number of rotations stored in object.

        """
        return self._matrix.shape[0]

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Transform':
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"Expected `matrix` to be numpy.ndarray got {matrix.__class__.__name__}")
        if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (4, 4):
            raise ValueError("Expected `matrix` to have shape (4, 4) or "
                             f"(N, 4, 4), got {matrix.shape}")
        if matrix.shape == (4, 4):
            matrix = matrix.reshape((1, 4, 4))
        n = matrix.shape[0]

        rot = matrix[:, :3, :3]
        trans = matrix[:, :3, 3]
        assert (matrix[:, 3, :] == np.array([0, 0, 0, 1])).all(), "Last row must be [0, 0, 0, 1]."

        rot = Rotation.from_matrix(rot).as_matrix()

        tf = np.repeat(np.eye(4).reshape((1, 4, 4)), n, 0)
        tf[:, :3, :3] = rot
        tf[:, :3, 3] = trans
        return cls(matrix=tf)

    @classmethod
    def from_quat_trans(cls, quat: np.ndarray, trans: np.ndarray) -> 'Transform':
        """
        Create a Transform object from the reference frame of quat, trans to a new one at position trans and oriented
        with quat.

        quat = x, y, z, w

        A reference frame.

        B reference frame.

        A_R_B = quat = orientation of B relative to A.

        A_D_B = trans = translation from A origin to B origin described in A.

        P_A point described in A.

        A_T_B = tf = Transform.from_pose(trans, quat).

        P_A = tf @ P_B.

        :param trans: translation in 3d.
        :param quat: quaternions.
        :return: Transform object.

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        """
        if not isinstance(quat, np.ndarray) and not isinstance(trans, np.ndarray):
            raise TypeError(f"Expected `quat` and `trans` to be numpy.ndarray got {quat.__class__.__name__} and"
                            f" {trans.__class__.__name__}")
        if trans.ndim not in [1, 2] or trans.shape[-1:] != (3,):
            raise ValueError("Expected `trans` to have shape (3,) or "
                             f"(N, 3), got {trans.shape}")
        if trans.shape == (3,):
            trans = trans.reshape((1, 3))
        n = trans.shape[0]

        rot = Rotation.from_quat(quat).as_matrix()
        tf = np.repeat(np.eye(4).reshape((1, 4, 4)), n, 0)
        tf[:, :3, :3] = rot
        tf[:, :3, 3] = trans
        return cls(tf)

    @classmethod
    def from_euler_trans(cls, seq: str, angles: np.ndarray, degrees: bool, trans: np.ndarray) -> 'Transform':
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

        if not isinstance(angles, np.ndarray) and not isinstance(trans, np.ndarray):
            raise TypeError(f"Expected `angles` and `trans` to be numpy.ndarray got {angles.__class__.__name__} and"
                            f" {trans.__class__.__name__}")
        if trans.ndim not in [1, 2] or trans.shape[-1:] != (3,):
            raise ValueError("Expected `trans` to have shape (3,) or "
                             f"(N, 3), got {trans.shape}")
        if trans.shape == (3,):
            trans = trans.reshape((1, 3))
        n = trans.shape[0]

        rot = Rotation.from_euler(seq=seq, angles=angles, degrees=degrees).as_matrix()
        tf = np.repeat(np.eye(4).reshape((1, 4, 4)), n, 0)
        tf[:, :3, :3] = rot
        tf[:, :3, 3] = trans
        return cls(tf)

    @classmethod
    def from_rot_mat_trans(cls, rot: np.ndarray, trans: np.ndarray) -> 'Transform':
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
        if not isinstance(rot, np.ndarray) and not isinstance(trans, np.ndarray):
            raise TypeError(f"Expected `rot` and `trans` to be numpy.ndarray got {rot.__class__.__name__} and"
                            f" {trans.__class__.__name__}")
        if trans.ndim not in [1, 2] or trans.shape[-1:] != (3,):
            raise ValueError("Expected `trans` to have shape (3,) or "
                             f"(N, 3), got {trans.shape}")
        if trans.shape == (3,):
            trans = trans.reshape((1, 3))
        n = trans.shape[0]

        new_rot = Rotation.from_quat(rot).as_matrix()
        tf = np.repeat(np.eye(4).reshape((1, 4, 4)), n, 0)
        tf[:, :3, :3] = new_rot
        tf[:, :3, 3] = trans
        return cls(tf)

    @classmethod
    def from_3_axis_rot_trans(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray, trans: np.ndarray) -> 'Transform':
        """

        :param x: unit vector of B x-axis expressed in A.
        :param y: unit vector of B y-axis expressed in A.
        :param z: unit vector of B z-axis expressed in A.
        :param trans: A_D_B = translation from A origin to B origin described in A.
        :return: The corresponding Transform object : A_tf_B
        """

        def check(var):
            if not isinstance(var, np.ndarray):
                raise TypeError(f"Expected `var`  to be numpy.ndarray got {var.__class__.__name__})")
            if var.ndim not in [1, 2] or var.shape[-1:] != (3,):
                raise ValueError("Expected `trans` to have shape (3,) or "
                                 f"(N, 3), got {var.shape}")
            if var.shape == (3,):
                return var.reshape((1, 3))
            return var

        trans = check(trans)
        x = check(x)
        y = check(y)
        z = check(z)

        n = trans.shape[0]

        # Inspired by https://stackoverflow.com/a/34392459/10054528
        a_r_b = np.stack([x, y, z], axis=2)  # A_R_B
        rot = Rotation.from_matrix(a_r_b).as_matrix()
        tf = np.repeat(np.eye(4).reshape((1, 4, 4)), n, 0)
        tf[:, :3, :3] = rot
        tf[:, :3, 3] = trans
        return cls(tf)

    def as_matrix(self) -> np.ndarray:
        if self._is_single:
            return self._matrix[0]
        else:
            return self._matrix

    def as_quat_trans(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        A reference frame.

        B reference frame.

        A_R_B = quat = orientation of B relative to A.

        A_D_B = trans = translation from A origin to B origin described in A.

        P_A point described in A.

        A_T_B = tf = Transform object such as P_A = Transform().from_pose(trans, quat) @ P_B.

        trans, quat = tf.get_pose().

        :return: quat(x, y, z, w), trans (x, y,z ) concatenated array.

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.                T_C_D = C_T_D
        """
        quat = Rotation.from_matrix(self.as_rot_mat()).as_quat()
        if self._is_single:
            return np.hstack((quat[0], self.as_trans()))
        else:
            return np.hstack((quat, self.as_trans()))

    def as_rot_quat(self) -> np.ndarray:
        """quat = x, y, z, w with scipy Rotation."""
        return Rotation.from_matrix(self.as_rot_mat()).as_quat()

    def as_rot_mat(self) -> np.ndarray:
        if self._is_single:
            return self._matrix[0, :3, :3]
        else:
            return self._matrix[:, :3, :3]

    def as_rot_euler(self, seq, degrees):
        """
        Cf doc scipy.spatial.transform.Rotation

        Return angle in the seq order
        Example for a rotation of pi/2 around z-axis if seq = zyx, will return np.array([pi/2, 0, 0])
        """
        return Rotation.from_matrix(self.as_rot_mat()).as_euler(seq=seq, degrees=degrees)

    def as_trans(self) -> np.ndarray:
        if self._is_single:
            return self._matrix[0, :3, 3]
        else:
            return self._matrix[:, :3, 3]

    def inv(self) -> 'Transform':
        r = self._matrix[:, :3, :3]
        rt = r.transpose((0, 2, 1))

        t = self.as_trans()
        t_rotated = (- rt @ t.reshape((-1, 3, 1))).squeeze()

        new_tf = np.repeat(np.eye(4).reshape((1, 4, 4)), len(r), 0)
        new_tf[:, :3, :3] = rt
        new_tf[:, :3, 3] = t_rotated
        return Transform(new_tf)


def _test_from_matrix_as_matrix():
    n = 1000
    quat = np.random.randn(n, 4)  # x, y, z , w

    # Test 1
    try:
        matrix = np.random.randn(1000, 4, 4)
        Transform.from_matrix(matrix).as_matrix()  # Must AssertionError
    except AssertionError:
        print("Input: Bad last row from matrix. Validated.")
    # Test 2
    matrix = np.eye(4)
    res = Transform.from_matrix(matrix).as_matrix()
    if (res == matrix).all():
        print("Eye(4, 4) Succeed.")

    # Test 3
    matrix = np.repeat(np.expand_dims(np.eye(4), 0), n, 0)
    res = Transform.from_matrix(matrix).as_matrix()
    if (res != matrix).all():
        print("Eye(N, 4, 4) Failed")

    # Test 4
    matrix = np.expand_dims(np.repeat(np.expand_dims(np.eye(4), 0), n, 0), 0)
    try:
        Transform.from_matrix(matrix).as_matrix()
    except ValueError:
        print("Shape (1, N, 4, 4): Succeed")

    # Test real transforms
    rot = np.random.randn(n, 3, 3)
    trans = np.random.randn(n, 3)
    matrix = np.repeat(np.expand_dims(np.eye(4), 0), n, 0)
    matrix[:, :3, :3] = rot
    matrix[:, :3, 3] = trans
    Transform.from_matrix(matrix).as_matrix()
    Transform.from_matrix(matrix).inv().as_matrix()
    Transform.from_quat_trans(quat, trans).as_matrix()

    x = np.random.randn(n, 3)
    y = np.random.randn(n, 3)
    z = np.random.randn(n, 3)
    trans = np.random.randn(n, 3)
    Transform.from_3_axis_rot_trans(x, y, z, trans).as_matrix()

    assert (Rotation.from_matrix(rot).as_matrix() == Transform.from_matrix(matrix).as_rot_mat()).all()

    # Why you must not compare rotations with quaternions
    msg = "Quaternion can sometimes have a sign difference, to compare rotation use matrices"
    try:
        assert (Rotation.from_matrix(rot).as_quat() == Transform.from_matrix(matrix).as_rot_quat()).all(), msg
    except AssertionError:
        print(msg)
    rot = np.array([[0.65277675, -0.399756, -1.45508672],
                    [0.20181084, 0.60089933, -1.41854686],
                    [-1.45746281, 0.70406115, -1.2892034]])
    matrix = np.array([[0.65277675, -0.399756, -1.45508672, -0.44544586],
                       [0.20181084, 0.60089933, -1.41854686, 0.37889411],
                       [-1.45746281, 0.70406115, -1.2892034, -0.13891746],
                       [0., 0., 0., 1.]])
    Rotation.from_matrix(rot).as_quat()
    Transform.from_matrix(matrix).as_rot_quat()


if __name__ == '__main__':
    _test_from_matrix_as_matrix()
