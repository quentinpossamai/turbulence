import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

import src.utils.paths as paths

plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif",
                     "font.sans-serif": ["Computer Modern Roman"]})
matplotlib.use('TKAgg')


def gen_video_habitat():
    import src.data.habitat.dataviz as dataviz
    # rgb1, z1, z1_hat = dataviz.get_image_from_generated_traj(9, 0, np.arange(40, 450))
    # rgb2, z2, z2_hat = dataviz.get_image_from_generated_traj(2, 14, np.arange(40, 250))
    # rgb3, z3, z3_hat = dataviz.get_image_from_generated_traj(6, 0, np.arange(400, 600))
    # pickle.dump([[rgb1, rgb2, rgb3],
    #              [z1, z2, z3],
    #              [z1_hat, z2_hat, z3_hat]], open(f'{dataviz.habitat.WD_PATH}data/habitat/video.pkl', 'wb'))

    [[rgb1, rgb2, rgb3],
     [z1, z2, z3],
     [z1_hat, z2_hat, z3_hat]] = pickle.load(open(f'{dataviz.habitat.WD_PATH}data/habitat/video.pkl', 'rb'))

    shape = [480, 640, 3]
    for rgb, i, z, z_hat in zip([rgb1, rgb2, rgb3],  #
                                [0, 1, 2],  #
                                [z1, z2, z3],  #
                                [z1_hat, z2_hat, z3_hat]):  #
        saving_path = f'{dataviz.habitat.WD_PATH}reports/IROS_01_03_2022/graphics/video/video{i}.mp4'
        fps = 20
        codec = cv2.VideoWriter_fourcc(*'H264')
        video = cv2.VideoWriter(saving_path, codec, float(fps), (shape[1], shape[0]), True)
        print(f'Video destination : {saving_path}')

        for j, img in tqdm(enumerate(rgb), total=len(rgb)):
            fig = plt.figure(figsize=(shape[1] / 100., shape[0] / 100.))
            ax = plt.axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis('off')
            symbol = r'$\hat{d}$'
            ax.annotate(f'd: {z[j]:0.2f}\n{symbol}: {z_hat[j]:0.2f}', (shape[1] - 10, 10), xycoords='data', size=8,
                        ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
            plt.close('all')
            video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
        video.release()
        cv2.destroyAllWindows()


def gen_video_euroc():
    import src.data.euroc_mav.loading2 as euroc
    dataset_name = 'euroc_mav'
    orga_path = f'{paths.WD_PATH}reports/IROS_01_03_2022/graphics/video/'
    z_gt = pickle.load(open(f'{orga_path}data_prediction/z_gt_fine_{dataset_name}.pkl', 'rb'))
    fa_gt = pickle.load(open(f'{orga_path}data_prediction/fa_gt_fine_{dataset_name}.pkl', 'rb'))
    z_hat = pickle.load(open(f'{orga_path}data_prediction/z_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_hat = pickle.load(open(f'{orga_path}data_prediction/fa_hat_fine_{dataset_name}.pkl', 'rb'))
    fa_gttest = pickle.load(open(f'{orga_path}data_prediction/fa_gttest_fine_{dataset_name}.pkl', 'rb'))

    dataset = euroc.EurocMAVDatasetHorizon2('test', 40, False, False, False, False)
    split = {'V1_01_easy/': [(i, e[1]) for i, e in enumerate(dataset.splits) if e[0] == 'V1_01_easy/'],
             'V1_02_medium/': [(i, e[1]) for i, e in enumerate(dataset.splits) if e[0] == 'V1_02_medium/'],
             'V1_03_difficult/': [(i, e[1]) for i, e in enumerate(dataset.splits) if e[0] == 'V1_03_difficult/']}

    shape = [480, 752, 3]
    for i, flight in zip([0, 1, 2], split.keys()):
        saving_path = f'{orga_path}video_euroc{i}.mp4'
        fps = 10
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(saving_path, codec, float(fps), (shape[1], shape[0]), True)
        print(f'Video destination : {saving_path}')

        for j, idx in tqdm(split[flight], total=len(split[flight])):
            fig = plt.figure(figsize=(shape[1] / 100., shape[0] / 100.))
            ax = plt.axes([0, 0, 1, 1])

            img_timestamp = dataset.data[flight]['cam0'][idx]
            img = np.array(imageio.imread(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{img_timestamp}.png"))
            color_frame = np.uint8(np.zeros(shape))
            for k in range(3):
                color_frame[..., k] = img

            ax.imshow(color_frame)
            ax.axis('off')
            # symbol = r'$\hat{d}$'
            ax.annotate(rf'$f_a$ GT: {fa_gt[j]:0.2f}' + '\n'
                                                        rf'$f_a$ Cheeseman Approx: {fa_gttest[j]:0.2f}' + '\n'
                                                                                                          rf'$f_a$ predicted (Ours): {fa_hat[j]:0.2f}',
                        (shape[1] - 10, 10), xycoords='data', size=8, ha='right', va='top',
                        bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
            plt.close('all')
            video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
        video.release()
        cv2.destroyAllWindows()


def gen_video_habitat_gru():
    import src.data.habitat.dataviz as dataviz
    orga_path = f'{paths.WD_PATH}reports/IROS_01_03_2022/graphics/'
    # rgb, _, _ = dataviz.get_image_from_generated_traj(24, 16, np.arange(140, 200))
    # pickle.dump(rgb, open(f'{dataviz.habitat.WD_PATH}data/habitat/video_gru.pkl', 'wb'))
    rgb = pickle.load(open(f'{dataviz.habitat.WD_PATH}data/habitat/video_gru.pkl', 'rb'))

    # rgb2, _, _ = dataviz.get_image_from_generated_traj(24, 16, np.arange(167, 171))
    rgb2 = pickle.load(open(f'{dataviz.habitat.WD_PATH}data/habitat/video_gru2.pkl', 'rb'))

    # z_gt = np.array(pickle.load(open(f'{paths.WD_PATH}figure6/z_gt.pkl', 'rb')))
    # z_hat = np.array(pickle.load(open(f'{paths.WD_PATH}figure6/z_hat.pkl', 'rb')))
    # z_hat2 = np.array(pickle.load(open(f'{paths.WD_PATH}figure6/z_hat2.pkl', 'rb')))
    fa_gt = np.array(pickle.load(open(f'{orga_path}figure6/fa_gt.pkl', 'rb')))
    fa_hat = np.array(pickle.load(open(f'{orga_path}figure6/fa_hat.pkl', 'rb')))
    fa_hat2 = np.array(pickle.load(open(f'{orga_path}figure6/fa_hat2.pkl', 'rb')))

    xx = np.arange(215, 275)
    xx2 = np.arange(39, len(fa_gt) + 39)
    # fig, ax = plt.subplots(figsize=(8, 3))
    # ax.plot(xx2[xx - 39], fa_gt[xx - 39], label='GT')
    # ax.plot(xx2[xx - 39], fa_hat[xx - 39], label='Ours')
    # ax.plot(xx2[xx - 39], fa_hat2[xx - 39], label='No GRU')
    # ax.grid()
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('fa (N)')
    # # ax.set_ylim([0.1, 1.2])
    # ax.axvline(240, c='black')
    # ax.axvline(246, c='black')
    # ax.legend()
    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.99, wspace=0., hspace=0.)
    # plt.show()

    shape = [480, 640, 3]
    saving_path = f'{dataviz.habitat.WD_PATH}reports/IROS_01_03_2022/graphics/video/video_gru.mp4'
    fps = 20
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(saving_path, codec, float(fps), (shape[1], shape[0]), True)
    print(f'Video destination : {saving_path}')

    for j, img in enumerate(tqdm(rgb)):
        fig = plt.figure(figsize=(shape[1] / 100., shape[0] / 100.))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis('off')
        ax.annotate((rf'f_a GT: {fa_gt[xx - 39][j]:0.5f}' + '\n'
                     rf'f_a (Ours): {fa_hat[xx - 39][j]:0.5f}' + '\n'
                     rf'f_a No GRU: {fa_hat2[xx - 39][j]:0.5f}'),
                    (shape[1] - 10, 10), xycoords='data', size=8, ha='right', va='top',
                    bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
        # plt.show()
        plt.close('all')
        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
    video.release()
    cv2.destroyAllWindows()


def main():
    # gen_video_habitat()
    # gen_video_euroc()
    gen_video_habitat_gru()


if __name__ == '__main__':
    main()
