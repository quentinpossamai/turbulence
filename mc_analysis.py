from util import Transform, Progress, DataFolder

from typing import Union, Iterable, List
# Union[np.ndarray, Iterable, int, float]
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle


def main():
    """
    This .py's goal is to extract drone pose (position + orientation) from MC (motion capture) data.
    """
    # Path extraction of the MC data
    print('Processing data path.')
    f = DataFolder(data_folder_name='data_drone2')

    for vol_number in [3, 4]:
        csv_paths = f.get_data_by_extension(extension='.csv', specific_folder=f.folders['raw'][vol_number])
        assert len(csv_paths) == 1
        csv_paths = csv_paths[0]

        # Extract poses of the drone from the motion capture measures
        print(f'Importing flight located in {csv_paths}\nCleaning raw data.')
        m = MCAnalysis(csv_paths)  # Importing and cleaning data
        data = m.get_data()
        pickle.dump(data, open(f.folders['intermediate'][vol_number] + 'mc_measure.pkl', 'wb'))  # Saving
        print('Computing poses from Motion Capture measures.')
        poses = m.get_pose()  # Extracting the pose from markers position
        print('Saving computed poses.')
        pickle.dump(poses, open(f.folders['intermediate'][vol_number] + 'mc_poses.pkl', 'wb'))  # Saving
        print('\n\n')


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

    def get_val(self, columns: List[str], index: Union[np.ndarray, Iterable, int, float]):
        return self.df[columns].iloc[index].to_numpy()

    def get_array(self, point_name: str, index: Union[np.ndarray, Iterable, int, float]):
        columns = [point_name + '_x', point_name + '_y', point_name + '_z']
        for e in columns:
            assert e in self.columns
        return self.df[columns].iloc[index].to_numpy()

    def get_center(self, index: Union[np.ndarray, Iterable, int, float]):
        return self.df[['y1_x', 'y1_y', 'y2_z']].iloc[index].to_numpy()


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
        columns = ['frame_id', 'sub_frame',
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
        data_raw: pd.DataFrame = pd.read_csv(path, sep=';', header=0, names=columns)

        # Drop headers
        data_raw = data_raw.iloc[4:, :].reset_index(drop=True)

        # Casting types
        dtypes = {e: ('float64' if i > 1 else 'int64') for i, e in enumerate(columns)}
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
        for e in columns[2:]:
            data_raw[[e]] = data_raw[[e]] * 1e-3

        self.data = MCData(data=data_raw)

    def get_data(self) -> MCData:
        """
        :return: Cleans imported data.
        """
        return self.data

    def get_pose(self) -> List[Transform]:
        """
        Saves the computed pose from markers position.
        """
        # figsize = (10, 12)
        # for e in ['b1', 'b2', 'b3', 'b4', 'x2', 'y1', 'y2']:
        #     fig, axs = plt.subplots(3, figsize=figsize)
        #     exec(f"self.self.df.df.plot(y='{e}_x', grid=True, legend=False, title='{e}_x', ax=axs[0])")
        #     exec(f"self.self.df.df.plot(y='{e}_y', grid=True, legend=False, title='{e}_y', ax=axs[1])")
        #     exec(f"self.self.df.df.plot(y='{e}_z', grid=True, legend=False, title='{e}_z', ax=axs[2])")
        #     plt.savefig(ABSOLUTE_PATH+f'plots/motion_capture_trajectory_comparison/{e}')
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

        # Get reference frame unit axis, compute initial position
        poses = []
        index = 0
        init_position = self.data.get_center(index)
        tmp = self.data.get_array('y2', index) - init_position
        x = tmp / np.linalg.norm(tmp)
        tmp = self.data.get_array('y1', index) - init_position
        z = tmp / np.linalg.norm(tmp)
        y = np.cross(z, x)

        # Save Transform object with a translation and an orientation, compute initial orientation
        init_x, init_y, init_z = x, y, z
        # Add current_tf_init to saved poses
        poses.append(Transform().from_euler(init_position, 'xyz', np.zeros(3)))
        _, init_orientation = poses[0].get_pose()

        init_mat = np.hstack([init_x.reshape((3, 1)), init_y.reshape((3, 1)), init_z.reshape((3, 1))])

        # Compute Transformation for all data
        p = Progress(len(self.data))
        for index, row in self.data.df.iterrows():
            # Get reference frame unit axis, compute actual position
            index: int
            position = self.data.get_center(index)
            tmp = self.data.get_array('y2', index) - position
            x = tmp / np.linalg.norm(tmp)
            tmp = self.data.get_array('y1', index) - position
            z = tmp / np.linalg.norm(tmp)
            y = np.cross(z, x)

            # Inspired by https://stackoverflow.com/a/34392459/10054528
            mat = np.hstack([x.reshape((3, 1)), y.reshape((3, 1)), z.reshape((3, 1))])
            init_d_current = position - init_position
            init_rot_current = np.transpose(mat @ np.linalg.pinv(init_mat -
                                                                 np.vstack([init_d_current.reshape((1, 3))
                                                                            for _ in range(3)])))
            # Add current_tf_init to saved poses
            poses.append(Transform().from_rot_matrix_n_trans_vect(init_d_current, init_rot_current))
            p.update_pgr()
        return poses


if __name__ == '__main__':
    main()
