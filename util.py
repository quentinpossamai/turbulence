from __future__ import print_function
import sys
import numpy as np
from typing import Union, Tuple, Optional, List
from scipy.spatial.transform import Rotation

import os
import re
import pathlib

ABSOLUTE_PATH = '/Users/quentin/phd/turbulence/'


class Progress(object):
    def __init__(self, max_iter: int, end_print: str = ''):
        """
        Initialise a progression printing object.
        :param max_iter: The maximum number of iteration.
        :param end_print: Print this string when the current iteration = max_iter.
        """
        self.max_iter = max_iter
        self.end_print = end_print
        self.iter = 0

    def update_pgr(self, iteration: int = None):
        """
        Update print of the progression percentage.
        :param iteration: If the progress went more than 1 it is possible to specify the actual iteration number.
        """
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

    def __repr__(self):
        return self.matrix.__repr__()

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
        """
        A reference frame.

        B reference frame.

        A_R_B = quat = orientation of B relative to A.

        A_D_B = trans = translation from A origin to B origin described in A.

        P_A point described in A.

        B_T_A = tf = Transform object such as P_B = Transform().from_pose(trans, quat) @ P_A.

        trans, quat = tf.get_pose().

        :return: trans, quat.

        Note: only work for 3D points, P_A = (x_A, y_A, z_A, 1) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        """
        tf = self.get_inv()
        quat = Rotation.from_matrix(tf.get_rot()).as_quat()
        trans = tf.get_trans()
        return trans, quat

    def from_pose(self, trans: np.ndarray, quat: np.ndarray) -> 'Transform':
        """
        Create a Transform object from the reference frame of (xyz, quat) to a new one at position xyz and oriented with
        quat.

        A reference frame.

        B reference frame.

        A_R_B = quat = orientation of B relative to A.

        A_D_B = trans = translation from A origin to B origin described in A.

        P_A point described in A.

        B_T_A = tf = Transform().from_pose(trans, quat).

        P_B = tf @ P_A.

        :param trans: translation in 3d.
        :param quat: quaternions.
        :return: Transform object.

        Note: only work for 3D points, P_A = (x_A, y_A, z_A, 1) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        """
        assert trans.shape == (3,)
        assert quat.shape == (4,)
        assert np.round(np.linalg.norm(quat), 3) == 1

        r = Rotation.from_quat(quat).as_matrix()
        r = np.transpose(r)
        t = - r @ trans
        tf = np.eye(4)
        tf[:3, :3] = r
        tf[:3, 3] = t
        self.matrix = tf
        return self

    def from_matrix(self, matrix: np.ndarray) -> 'Transform':
        self.matrix = matrix
        return self

    def from_euler(self, trans: np.ndarray, seq: str,
                   angles: np.ndarray, degrees: Optional[bool] = False) -> 'Transform':
        """
        Return a Transform object from a translation, an angle convention, and 3 Euler angles.
        :param trans: The translation of the Transform, must be of length 3
        :param seq: Angle convention of the Euler angles, can be 'xyz' or any other order
        :param angles: An array of the 3 Euler angles in the order of the convention
        :param degrees: True if the unit of the Euler angles is degree or False for radian
        :return:
        """
        assert angles.shape == (3,)
        return self.from_pose(trans=trans, quat=Rotation.from_euler(seq=seq, angles=angles, degrees=degrees).as_quat())

    def from_rot_matrix_n_trans_vect(self, trans: np.ndarray, rot: np.ndarray) -> 'Transform':
        """
        A reference frame.

        B reference frame.

        :param trans: A_D_B = translation from A origin to B origin described in A.

        :param rot: A_R_B = numpy 3x3 rotation matrix, orientation of B relative to A.

        P_A point described in A.

        :return: B_T_A = tf = Transform object such as
        P_B = Transform().from_rot_matrix_n_trans_vect(trans, quat) @ P_A.

        Note: only work for 3D points, P_A = (x_A, y_A, z_A, 1) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        Return the Transform object from a rotation matrix and a translation vector.
        """
        quat = Rotation.from_matrix(rot).as_quat()
        self.from_pose(trans, quat)
        return self


class DataFolder(object):
    """
    Manages the data organisation.
    """
    def __init__(self, data_folder_name: str):
        """
        .folders attribute is a dict with folders[purpose_folder_names][data_folder_number]
        purpose_folder_names: str. It is defined below.
        :param data_folder_name the name of the data folder to use.
        """
        self.workspace_path = ABSOLUTE_PATH
        self.data_path = ABSOLUTE_PATH + data_folder_name + '/'

        # Add "purpose" folders
        purpose_folder_names = ['raw', 'intermediate', 'results' ,'plots']
        self.folders = {e: self.data_path + e + '/' for e in purpose_folder_names}

        # Add all of data folder in each purpose folders
        raw_folder_to_imitate = purpose_folder_names[0]  # 'raw'
        dirs = sorted(next(os.walk(self.data_path + raw_folder_to_imitate + '/'))[1])
        for folder in purpose_folder_names:
            self.folders[folder] = {i: self.folders[folder] + e + '/' for i, e in enumerate(dirs)}
            self.folders[folder][''] = self.data_path + folder + '/'  # Add racine into dict

        # Create all the folders
        self.create_folder(self.folders)

        # List all the folders to sort
        folders_to_sort = sorted([e[0] for e in os.walk(self.data_path)])

        # Sort by extension
        pattern = '(?=\\.)(.*)'
        raw_sorting_dict = {}
        for fold in folders_to_sort:
            for e in sorted(os.listdir(fold)):
                a = re.search(pattern, e)
                if a is None:
                    continue
                else:
                    a = a.group(0)
                if a not in raw_sorting_dict:
                    raw_sorting_dict[a] = []
                raw_sorting_dict[a].append(fold + '/' + e)

        # Sort path
        for key in raw_sorting_dict:
            raw_sorting_dict[key] = sorted(raw_sorting_dict[key])
        self.raw_sorting_dict = raw_sorting_dict

    def create_folder(self, d: Union[list, dict]):
        """
        Creates folders given a list or dict of paths.
        Can be a N nested dict/list of dict/list of paths.
        :param d: The N nested dict/list of dict/list of paths.
        """
        tmp = d

        def f(x):
            return pathlib.Path(x).mkdir(parents=True, exist_ok=True)

        if isinstance(d, dict):
            tmp = d.values()
        for v in tmp:
            if isinstance(v, (dict, list)):
                self.create_folder(v)
            else:
                f(v)

    def get_data_by_extension(self, extension: str, specific_folder: Union[str, List[str]] = None) -> List[str]:
        """
        :param extension: The name of the extension.
        :param specific_folder:
        :return: List of paths of all raw data files that are named with this extension.
        """
        assert extension in self.raw_sorting_dict
        if specific_folder is None:
            return self.raw_sorting_dict[extension]
        else:
            to_keep = []
            if type(specific_folder) is str:
                specific_folder = [specific_folder]
            nominee = self.raw_sorting_dict[extension].copy()
            for fold in specific_folder:
                for e in nominee:
                    e_folder = '/'.join(e.split('/')[:-1]) + '/'
                    if e_folder == fold:
                        to_keep.append(e)
            return to_keep

def object_analysis(obj):
    """
    Returns all attributes and methods of an object and classify them by
    (variable, method, public, private, protected).
    :param obj: object. The object to by analysed.
    :return: Dict[str, Dict[str, List]]
        Returns a dict of 2 dicts :
        -'variable'= dict
        -'method'= dict
        and for each of those dicts 3 lists:
        public, private, or protected.
    """
    res = {'variable': {'public': [], 'private': [], 'protected': []},
           'method': {'public': [], 'private': [], 'protected': []}}
    var_n_methods = sorted(dir(obj))
    for e in var_n_methods:
        try:
            if callable(getattr(obj, e)):
                attribute_or_method = 'method'
            else:
                attribute_or_method = 'variable'
            if len(e) > 1:
                if e[0] + e[1] == '__':
                    res[attribute_or_method]['private'].append(e)
                    continue
            if e[0] == '_':
                res[attribute_or_method]['protected'].append(e)
            else:
                res[attribute_or_method]['public'].append(e)
        except AttributeError:
            print('Attribute : {} of object {} is listed by dir() but cannot be accessed.'.format(e, type(obj)))
    return res

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
