from __future__ import print_function

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Union, Tuple, Optional, List, Iterable
import sys
import os
import re
import pathlib
import time
import pickle
import pandas as pd

ABSOLUTE_PATH = '/Users/quentin/phd/turbulence/'


class Progress(object):
    def __init__(self, max_iter: int, start_print: str = None, end_print: str = None):
        """
        Initialise a progression printing object.
        :param max_iter: The maximum number of iteration.
        :param start_print: Print this string when initializing Progress.
        :param end_print: Print this string when the current iteration = max_iter.
        """
        self.max_iter = max_iter
        self.start_print = start_print
        self.end_print = end_print
        self.iter = 0
        self.initial_time = time.time()

        if start_print is not None:
            print(start_print)

    def update_pgr(self, iteration: int = None):
        """
        Update print of the progression percentage.
        :param iteration: If the progress went more than 1 it is possible to specify the actual iteration number.
        """
        if iteration is not None:
            self.iter = iteration
        # Progression
        print(f'\rProgression : {(self.iter + 1) * 100 / self.max_iter:0.02f}% | '
              f'Time passed : {time.time() - self.initial_time:.03f}s', end='')
        sys.stdout.flush()
        if self.iter + 1 == self.max_iter:
            if self.end_print is not None:
                print(self.end_print)
            else:
                print('')
        self.iter += 1


class Transform(object):
    def __init__(self):
        self.matrix: np.ndarray = np.eye(4)

    def __matmul__(self, multiple: Union['Transform', np.ndarray]) -> Union['Transform', np.ndarray]:
        if type(multiple) is Transform:
            return Transform().from_matrix(self.matrix @ multiple.get_matrix())
        else:  # type(multiple) is np.ndarray
            if multiple.shape in [(4,), (4, 1)]:
                return self.matrix @ multiple
            elif multiple.shape in [(3,), (3, 1)]:
                return (self.matrix @ np.append(multiple, 1))[:3]

    def __repr__(self):
        return self.matrix.__repr__()

    def get_rot(self) -> np.ndarray:
        return self.matrix[:3, :3]

    def get_trans(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def inv(self) -> 'Transform':
        r = self.get_rot()
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

        B_T_A = tf = Transform object such as P_B = Transform().from_pose(trans, quat) @ P_A.

        trans, quat = tf.get_pose().

        :return: trans, quat.

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A, 1) while applying '@' operator.

        C reference frame.

        D reference frame.

        C_T_D = C_T_A @ A_T_B @ B_T_D.
        """
        tf = self.inv()
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

        Note: only work for 3D points_name, P_A = (x_A, y_A, z_A, 1) while applying '@' operator.

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

        :param trans: B_D_A = translation from B origin to A origin described in B.

        :param rot: B_R_A = A 3x3 rotation matrix, orientation of A relative to B.

        P_A point described in A.

        :return: B_T_A = tf = Transform object such as
        P_B = Transform().from_rot_matrix_n_trans_vect(trans, quat) @ P_A.

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

    def from_trans_n_axis(self, trans: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> 'Transform':
        """

        :param x: unit vector of B x-axis expressed in A.
        :param y: unit vector of B y-axis expressed in A.
        :param z: unit vector of B z-axis expressed in A.
        :param trans: A_D_B = translation from A origin to B origin described in A.
        :return: The corresponding Transform object : B_tf_A
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

        # Add "purpose" folders, first purpose folder will be the reference to construct the others.
        purpose_folder_names = ['raw', 'raw_python', 'intermediate', 'results', 'plots']

        # self.folders[purpose_folder_names][vol_number] folders can be used like that
        self.folders = {e: self.data_path + e + '/' for e in purpose_folder_names}

        # Add all of data folder in each purpose folders
        raw_folder_to_imitate = purpose_folder_names[0]  # 'raw'
        if not os.path.isdir(self.data_path + raw_folder_to_imitate + '/'):
            raise FileNotFoundError(f"Folder 'raw' not defined at {self.data_path + raw_folder_to_imitate + '/'}")
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

    def get_files_paths(self, extension: str = None, specific_folder: Union[str, List[str]] = None,
                        filename_begin_with: str = None) -> List[str]:
        """
        :param extension: The name of the extension.
        :param specific_folder: Path or list of paths to match the file's folder's path.
        :param filename_begin_with: Beginning of the filename.
        :return: List of paths of all raw data files that are named with this extension.
        """
        assert extension in self.raw_sorting_dict
        if extension is None:
            to_keep = []
            for ext, paths in self.raw_sorting_dict.items():
                to_keep += paths
        else:
            to_keep = self.raw_sorting_dict[extension]

        # specific_folder search
        to_look = to_keep.copy()
        if specific_folder is None:
            to_keep = to_look.copy()
        else:
            to_keep = []
            if type(specific_folder) is str:
                specific_folder = [specific_folder]
            nominee = to_look.copy()
            for fold in specific_folder:
                for file_path in nominee:
                    if fold + file_path[len(fold):] == file_path:
                        to_keep.append(file_path)

        # filename_begin_with search
        to_look = to_keep.copy()
        if filename_begin_with is not None:
            for file_path in to_look:
                filename = get_file_name(file_path)
                res = re.search(filename_begin_with, filename)
                if res is None:
                    to_keep.remove(file_path)
            if not to_keep:
                print('Nothing found.')
                raise FileNotFoundError
            return to_keep
        else:
            return to_keep

    def get_unique_file_path(self, extension: str = None, specific_folder: Union[str, List[str]] = None,
                             filename_begin_with: str = None) -> str:
        """
        :param extension: The name of the extension.
        :param specific_folder: Path or list of paths to match the file's folder's path.
        :param filename_begin_with: Beginning of the filename.
        :return: List of paths of all raw data files that are named with this extension.
        """
        to_keep = self.get_files_paths(extension, specific_folder, filename_begin_with)
        if len(to_keep) > 1:
            print('Too much files founded :')
            print(to_keep)
            raise FileNotFoundError
        else:
            return to_keep[0]

    def pickle_load_file(self, extension: str = None, specific_folder: Union[str, List[str]] = None,
                         filename_begin_with: str = None, pickle_was_python2: bool = False):
        """
        :param extension: The name of the extension.
        :param specific_folder: Path or list of paths to match the file's folder's path.
        :param filename_begin_with: Beginning of the filename.
        :param pickle_was_python2: If the file was pickle using python2 or not.
        :return: List of paths of all raw data files that are named with this extension.
        """

        to_load = self.get_unique_file_path(extension, specific_folder, filename_begin_with)
        if pickle_was_python2:
            return pickle.load(open(to_load, 'rb'), encoding='latin1')
        else:
            return pickle.load(open(to_load, 'rb'))


def get_file_name(file_path: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Return the files names from the files paths.
    :param file_path: The paths of the files.
    :return: The files names.
    """
    if type(file_path) == str:
        assert file_path[-1] != '/', "file_path end with '/'. It is a folder path."
        return file_path.split('/')[-1]
    else:  # file_path is a list
        for e in file_path:
            assert e[-1] != '/', "file_path end with '/'. It is a folder path."
        return [e.split('/')[-1] for e in file_path]


def get_folder_path(file_path: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove the file name in the file path to ge the path of the folder containing the file.
    :param file_path: The path of the file.
    :return: Folder path.
    """
    if type(file_path) == str:
        assert file_path[-1] != '/', "file_path end with '/'. It is a folder path."
        return '/'.join(file_path.split('/')[:-1]) + '/'
    else:  # file_path is a list
        for e in file_path:
            assert e[-1] != '/', "file_path end with '/'. It is a folder path."
        return ['/'.join(e.split('/')[:-1]) + '/' for e in file_path]


def merge_two_arrays(array1: Union[np.ndarray, Iterable, int, float],
                     array2: Union[np.ndarray, Iterable, int, float]) -> Tuple[Union[List[int], int],
                                                                               Union[List[int], int]]:
    """
    From two sorted arrays of different length and uniques values : array1 and array2.
    :param array1: An array of values to be compared to array2.
    :param array2: An array of values to be compared to array1.
    :return: Return one indices array for each that indicate the closest value.
    """
    is_array1_iterable = is_iterable(array1)
    is_array2_iterable = is_iterable(array2)

    input_vars = {'array1': array1, 'array2': array2}
    for name, arr in input_vars.items():
        if isinstance(arr, int) or isinstance(arr, float):
            input_vars[name] = np.array([input_vars[name]])
        elif isinstance(arr, pd.Series):
            input_vars[name] = arr.to_numpy()
        elif not isinstance(arr, np.ndarray):
            input_vars[name] = np.asarray(input_vars[name])
    array1 = input_vars['array1']
    array2 = input_vars['array2']

    # Hypothesis verification
    assert len(array1) > 0, 'array1 must not be empty'
    assert len(array1) == len(np.unique(array1)), 'array1 must have unique elements.'
    assert array1.any() == np.sort(array1).any(), 'array1 must be sorted.'

    assert len(array2) > 0, 'array2 must not be empty'
    assert len(array2) == len(np.unique(array2)), 'array2 must have unique elements.'
    assert array2.any() == np.sort(array2).any(), 'array2 must be sorted.'

    # Initialisation
    little_ids = []
    big_ids = []
    if len(array1) < len(array2):
        case = 0
        little_array = array1.copy()
        is_little_iterable = is_array1_iterable
        big_array = array2.copy()
        is_big_iterable = is_array2_iterable
    else:
        case = 1
        little_array = array2.copy()
        is_little_iterable = is_array2_iterable
        big_array = array1.copy()
        is_big_iterable = is_array1_iterable
    little_array_remaining_id = {i: val for i, val in enumerate(little_array)}  # The remaining idx not taken
    big_array_remaining_id = {i: val for i, val in enumerate(big_array)}  # The remaining idx not taken

    # Algorithm begin
    for little_id, e1 in enumerate(little_array):
        # Find the closer to e1 in big_array
        big_id, big_val = find_nearest(big_array, e1)
        if big_id in big_array_remaining_id:
            little_ids.append(little_id)
            big_ids.append(big_id)
            del little_array_remaining_id[little_id]
            del big_array_remaining_id[big_id]
        else:
            diff1 = np.abs(little_array[little_id - 1] - big_val)
            diff2 = np.abs(little_array[little_id] - big_val)
            if diff1 > diff2:
                del little_ids[-1]  # remove little_id - 1
                little_array_remaining_id[little_id - 1] = little_array[little_id - 1]
                little_ids.append(little_id)
                del little_array_remaining_id[little_id]

    assert len(little_ids) > 0, 'No correspondence founded'
    assert len(big_ids) > 0, 'No correspondence founded'

    if (not is_little_iterable) or (not is_big_iterable):
        little_ids = little_ids[0]
        big_ids = big_ids[0]

    if case == 0:
        return little_ids, big_ids
    else:
        return big_ids, little_ids


def plane_equation(p1: Union[np.ndarray, list], p2: Union[np.ndarray, list],
                   p3: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the equation of 2D plane from 3D points_name.
    :param p1: Point in 3D space different from p2 and p3.
    :param p2: Point in 3D space different from p1 and p3.
    :param p3: Point in 3D space different from p2 and p1.
    :return: The 4 parameters a, b, c, d of a plane equation in an array form.
    """
    # Type correction
    input_vars = {'p1': p1, 'p2': p2, 'p3': p3}
    for name, arr in input_vars.items():
        if isinstance(arr, list) or isinstance(arr, float):
            input_vars[name] = np.array(input_vars[name])
        elif not isinstance(arr, np.ndarray):
            raise TypeError('Points type must be list or np.ndarray')
    p1 = input_vars['p1']
    p2 = input_vars['p2']
    p3 = input_vars['p3']

    # Vector P
    p2_p1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    p3_p1 = (p3 - p1) / np.linalg.norm(p3 - p1)
    plane_normal = np.cross(p3_p1, p2_p1)
    plane_normal /= np.linalg.norm(plane_normal)
    d = - plane_normal @ p1
    return np.hstack([plane_normal, d])


def find_nearest(array, value):
    """
    Find nearest index and value into an array according to an input value. Nearest i.e with the smaller absolute value.
    :param array: The array to be analysed.
    :param value: The input value.
    :return: index and value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def is_iterable(obj) -> bool:
    """
    Test if obj is an iterable or not.
    :param obj: An object to be tested.
    :return: True or False.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


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


def angle_arccos(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    return np.arccos((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))
