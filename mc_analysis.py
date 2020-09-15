import util

from typing import Union, Iterable, List, Dict
# Union[np.ndarray, Iterable, int, float]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


def main():
    """
    This .py's goal is to extract drone pose (position + orientation) from MC (motion capture) data.
    """
    # Path extraction of the MC data
    print('Processing data path.')
    f = util.DataFolder(data_folder_name='data_drone2')

    for vol_number in [0, 1, 2, 3, 4]:
        csv_path = f.get_unique_file_path(extension='.csv', specific_folder=f.folders['raw'][vol_number])

        # Extract poses of the drone from the motion capture measures
        print(f'Importing flight located in {csv_path}\nCleaning raw data.')
        m = MCAnalysis(csv_path)  # Importing and cleaning data
        data = m.get_data()
        pickle.dump(data.df, open(f.folders['raw_python'][vol_number] + 'mc_measure.pkl', 'wb'))  # Saving
        print('Computing poses from Motion Capture measures.')
        poses = m.get_pose()  # Extracting the pose from markers position
        print('Saving computed poses.')
        pickle.dump(poses, open(f.folders['raw_python'][vol_number] + 'mc_poses.pkl', 'wb'))  # Saving
        print('\n\n')


class MCAnalysis(object):
    """
    MCAnalysis regroup the methods to extract the pose from the motion capture data.
    """

    def __init__(self, path: str):
        """
        Load raw MC data. z is the vertical axis oriented from down to top.
        :param path: path of the .csv file of mc data.
        """
        # Naming columns and removing headers
        self.columns = ['frame_id', 'sub_frame',
                        'b1_x', 'b1_y', 'b1_z',
                        'b2_x', 'b2_y', 'b2_z',
                        'b3_x', 'b3_y', 'b3_z',
                        'b4_x', 'b4_y', 'b4_z',
                        'y1_x', 'y1_y', 'y1_z',
                        'y2_x', 'y2_y', 'y2_z',
                        'x2_x', 'x2_y', 'x2_z',
                        'clapet_inf1_x', 'clapet_inf1_y', 'clapet_inf1_z',
                        'clapet_inf2_x', 'clapet_inf2_y', 'clapet_inf2_z',
                        'clapet_sup_2_x', 'clapet_sup_2_y', 'clapet_sup_2_z']
        self.points_name = ['b1', 'b2', 'b3', 'b4', 'y1', 'y2', 'x2']

        data_raw: pd.DataFrame = pd.read_csv(path, sep=';', header=0, names=self.columns)

        # Drop headers
        data_raw = data_raw.iloc[4:, :].reset_index(drop=True)

        # Casting types
        dtypes = {e: ('float64' if i > 1 else 'int64') for i, e in enumerate(self.columns)}
        data_raw = data_raw.astype(dtypes)

        # Frame id start at 0
        data_raw[['frame_id']] = data_raw[['frame_id']] - 1

        # Cutting non-usable data
        filename = path.split('/')[-1]
        to_cut = {'VolAvecPoubelle03.csv': 5000,
                  'VolAvecPoubelle04.csv': 4000}
        if filename in to_cut:
            data_raw.drop(list(data_raw.index)[to_cut[filename]:], inplace=True)

        # From mm to m
        for e in self.columns[2:]:
            data_raw[[e]] = data_raw[[e]] * 1e-3

        self.data = MCData(data=data_raw)
        self.drone_tf_o = np.zeros((4, 4))  # Will be defined in self.c_reference_frame

    def get_data(self) -> 'MCData':
        """
        :return: Cleans imported data.
        """
        return self.data

    def object_reference_frame(self) -> Dict[str, np.ndarray]:
        """
        Define the reference frame of the drone, C.

        :return: All the measured points_name expressed in C.
        """
        # Definition of C the reference frame of the drone.
        # First it's origin C through the computing of OC. With this definition it is almost the center of mass.
        oc = np.zeros(3)
        for i in range(4):
            oc += self.data.get_point(f'b{i + 1}', 0)
        oc = oc / 4

        # bi's plane equation
        bis_eq = util.plane_equation(self.data.get_point('b1', 0),
                                     self.data.get_point('b2', 0),
                                     self.data.get_point('b3', 0))
        plane_normal = bis_eq[0:3]

        # x axis definition
        b1 = self.data.get_point('b1', 0)
        y2 = self.data.get_point('y2', 0)
        y2proj = (y2 - b1) - ((y2 - b1) @ plane_normal) * plane_normal + b1
        x = (y2proj - oc) / np.linalg.norm(y2proj - oc)

        # z axis definition
        z = plane_normal / np.linalg.norm(plane_normal)

        # y axis definition in O
        y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))

        # Compute ai'
        self.drone_tf_o = util.Transform().from_trans_n_axis(trans=oc, x=x, y=y, z=z)
        res = {}
        for point_name in self.points_name:
            # TODO delete prints
            # print(f'{point_name} in O : {self.data.get_point(point_name, 0)}')
            res[point_name] = self.drone_tf_o @ self.data.get_point(point_name, 0)
            # print(f'{point_name} in drone : {res[point_name]}')

        # TITLE : Plotting ai and ai'
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i, point_name in enumerate(self.points_name):
        #     # R0
        #     point = self.data.get_point(point_name, 0)
        #     ax.scatter(xs=point[0], ys=point[1], zs=point[2], c='tab:blue')
        #     ax.text(x=point[0], y=point[1], z=point[2], s=point_name)
        #     # Drone
        #     point = res[point_name]
        #     ax.scatter(xs=point[0], ys=point[1], zs=point[2], c='tab:orange')
        #     ax.text(x=point[0], y=point[1], z=point[2], s=point_name)
        #     if i == 0:
        #         ax.scatter(xs=point[0], ys=point[1], zs=point[2], c='tab:blue', label='R0')
        #         ax.scatter(xs=point[0], ys=point[1], zs=point[2], c='tab:orange', label='Drone')
        #
        # # y2 projected
        # point = y2proj
        # point_name = 'y2proj'
        # ax.scatter(xs=point[0], ys=point[1], zs=point[2], c='tab:green')
        # ax.text(x=point[0], y=point[1], z=point[2], s=point_name)
        #
        # # C
        # point = oc
        # point_name = 'C'
        # ax.scatter(xs=point[0], ys=point[1], zs=point[2], c='tab:green')
        # ax.text(x=point[0], y=point[1], z=point[2], s=point_name)
        #
        # # Plane normal
        # ax.quiver(oc[0], oc[1], oc[2], plane_normal[0], plane_normal[1], plane_normal[2], length=0.1)
        #
        # ax.view_init(azim=0, elev=90)
        # ax.legend()
        # plt.show()

        return res

    def get_pose(self) -> List[util.Transform]:
        """
        Saves the computed pose from markers position.
        """
        figsize = (10, 12)
        # for e in ['b1', 'b2', 'b3', 'b4', 'x2', 'y1', 'y2']:
        #     fig, axs = plt.subplots(3, figsize=figsize)
        #     exec(f"self.self.df.df.plot(y='{e}_x', grid=True, legend=False, title='{e}_x', ax=axs[0])")
        #     exec(f"self.self.df.df.plot(y='{e}_y', grid=True, legend=False, title='{e}_y', ax=axs[1])")
        #     exec(f"self.self.df.df.plot(y='{e}_z', grid=True, legend=False, title='{e}_z', ax=axs[2])")
        #     # plt.savefig(ABSOLUTE_PATH+f'plots/motion_capture_trajectory_comparison/{e}')
        # plt.show()

        # for axis in ['x', 'y', 'z']:
        #     fig = plt.figure(figsize=figsize)
        #     plt.title(axis)
        #     # xx = np.linspace(0, len(self.df.df) - 1, len(self.df.df))
        #     end_x = 3000
        #     xx = np.linspace(0, end_x - 1, end_x)
        #     for e in ['b1', 'b2', 'b3', 'b4', 'x2', 'y1', 'y2']:
        #         plt.plot(xx, self.data.df[e + '_' + axis][0:end_x], label=e)
        #         plt.legend()
        # plt.show()

        poses = []

        # Compute Transformation between drone and motion capture reference frame for all data
        # Inspired by :
        # http://nghiaho.com/?page_id=671&fbclid=IwAR3ss4avz2OyZmGeQRe9ZhDFF5slMKDQa3LLaSGZttcggvzkCqBBjM7MKvA

        # Initialisation
        ai_mat = np.zeros((len(self.points_name), 3))
        ai_prime_mat = np.zeros((len(self.points_name), 3))
        p = util.Progress(len(self.data))

        # Get ai'
        ai_prime = self.object_reference_frame()
        for i, point_name in enumerate(self.points_name):
            ai_prime_mat[i, :] = ai_prime[point_name]
        # Get ai' centroid
        centroid_ai_prime = np.mean(ai_prime_mat, axis=0)

        # Compute centroid for ai and then compute rotation matrix R and translation array T
        for index in self.data.df.index:
            # Get ai and their centroid
            for i, point_name in enumerate(self.points_name):
                ai_mat[i, :] = self.data.get_point(point_name, index)
            centroid_ai = np.mean(ai_mat, axis=0)

            # Compute rotation matrix
            # h = np.transpose(ai_mat - centroid_ai) @ (ai_prime_mat - centroid_ai_prime)
            h = np.transpose(ai_mat) @ ai_prime_mat
            u, s, v = np.linalg.svd(h)
            drone_r_origin = v.T @ u.T
            if np.linalg.det(drone_r_origin) < 0:
                u, s, v = np.linalg.svd(drone_r_origin)
                v[:, 2] = -1 * v[:, 2]
                drone_r_origin = v.T @ u.T

            # Compute translation array
            drone_t_origin = centroid_ai_prime - drone_r_origin @ centroid_ai

            ai_prime_mat - (np.transpose(drone_r_origin @ np.transpose(ai_mat)) + drone_t_origin)
            p.update_pgr()
            pass
        return poses


class MCData(object):
    def __init__(self, data: pd.DataFrame):
        self.columns = ['frame_id', 'sub_frame',
                        'b1_x', 'b1_y', 'b1_z',
                        'b2_x', 'b2_y', 'b2_z',
                        'b3_x', 'b3_y', 'b3_z',
                        'b4_x', 'b4_y', 'b4_z',
                        'y1_x', 'y1_y', 'y1_z',
                        'y2_x', 'y2_y', 'y2_z',
                        'x2_x', 'x2_y', 'x2_z',
                        'clapet_inf1_x', 'clapet_inf1_y', 'clapet_inf1_z',
                        'clapet_inf2_x', 'clapet_inf2_y', 'clapet_inf2_z',
                        'clapet_sup_2_x', 'clapet_sup_2_y', 'clapet_sup_2_z']
        self.df = data

    def __len__(self):
        return len(self.df)

    def get_val(self, columns: List[str], index: Union[np.ndarray, Iterable, int, float]) -> np.ndarray:
        """
        Get the values of df given column names and indexes.
        :param columns: The columns in which the values are. Possible columns are :

        ['frame_id', 'sub_frame',
         'b1_x', 'b1_y', 'b1_z',
         'b2_x', 'b2_y', 'b2_z',
         'b3_x', 'b3_y', 'b3_z',
         'b4_x', 'b4_y', 'b4_z',
         'y1_x', 'y1_y', 'y1_z',
         'y2_x', 'y2_y', 'y2_z',
         'x2_x', 'x2_y', 'x2_z',
         'clapet_inf1_x', 'clapet_inf1_y', 'clapet_inf1_z',
         'clapet_inf2_x', 'clapet_inf2_y', 'clapet_inf2_z',
         'clapet_sup_2_x', 'clapet_sup_2_y', 'clapet_sup_2_z']

        :param index: The indexes in which the values are.
        :return: The values in a numpy array.
        """
        return self.df[columns].iloc[index].to_numpy()

    def get_point(self, point_name: str, index: Union[np.ndarray, Iterable, int, float]) -> np.ndarray:
        """
        Get the 3 coordinates of the point in self.df giving the point name and it's index in the pd.DataFrame
        :param point_name: The name of the point. Possible point name are :

        ['b1', 'b2', 'b3', 'b4', 'y1', 'y2', 'x2', 'clapet_inf1', 'clapet_inf2', 'clapet_sup_2']

        :param index: The index in df.
        :return: The coordinates of the points_name in the form of a numpy array of length 3.
        """
        columns = [point_name + '_x', point_name + '_y', point_name + '_z']
        for e in columns:
            assert e in self.columns
        return self.df[columns].iloc[index].to_numpy()


if __name__ == '__main__':
    main()
