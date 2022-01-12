import pickle

from src.utils.paths import WD_PATH

LOCOBOT_DATA_PATH = f'{WD_PATH}data/locobot/'

TRAJ_LIST = ['2021-01-26-06-53-32_color-depth-tf-part1/',
             '2021-01-26-06-53-32_color-depth-tf-part2/',
             '2021-01-26-06-53-32_color-depth-tf-part3/',
             '2021-01-26-06-53-32_color-depth-tf-part4/',
             '2021-01-26-06-53-32_color-depth-tf-part5/',
             '2021-01-26-06-53-32_color-depth-tf-part6/',
             '2021-01-26-06-53-32_color-depth-tf-part7/',
             '2021-01-26-06-53-32_color-depth-tf-part8/',
             '2021-01-26-07-09-21_color-depth-tf-part1/',
             '2021-01-26-07-09-21_color-depth-tf-part2/',
             '2021-03-18-11-07-50_color-depth-tf/']


def loading_traj(traj):
    data = pickle.load(open(f'{LOCOBOT_DATA_PATH}{traj}data.pkl', 'rb'))
    return data


def main():
    for i, traj in enumerate(TRAJ_LIST):
        print(f'{i} | traj: {traj}')
        loading_traj(traj)
        print('Success\n')


if __name__ == '__main__':
    main()
