import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm

import src.data.euroc_mav.loading as euroc
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
from src.drone_models.residual.gru_cnn import GRUCNNResidual
from src.utils.paths import mkdir_if_not_exist, ANALYSIS_DATA_VIS_EUROC_MAV, ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING

matplotlib.use('TKAgg')
mkdir_if_not_exist([ANALYSIS_DATA_VIS_EUROC_MAV])


def plot_y_x():
    flight = euroc.FLIGHT_LIST[0]
    # list[dict_keys(['timestamp', 'embedding', 'MSE', 'GradMSE'])]
    depth_emb = pickle.load(open(f"{euroc.INTERMEDIATE_DATASET_PATH}{flight}depth_embedding.pkl", "rb"))
    depth_emb = sorted(depth_emb, key=lambda x: x["timestamp"])

    # list[dict_keys(['timestamp', 'embedding', 'MSE', 'CrossProduct'])]
    norm_emb = pickle.load(open(f"{euroc.INTERMEDIATE_DATASET_PATH}{flight}normal_embedding.pkl", "rb"))
    norm_emb = sorted(norm_emb, key=lambda x: x["timestamp"])

    depth_grad_mse = np.array([e["GradMSE"] for e in depth_emb])
    depth_mse = np.array([e["MSE"] for e in depth_emb])
    norm_cross_prod = np.array([e["CrossProduct"] for e in norm_emb])
    norm_psnr = np.array([e["MSE"] for e in norm_emb])
    depth_psnr = 10 * np.log10(4 / depth_mse)
    norm_psnr = 10 * np.log10(4 / norm_psnr)

    # 1-DEPTH EMBEDDINGS
    # 1.1-Depth grad mse histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(depth_grad_mse, bins=100)
    ax.grid()
    ax.set_xlabel("Grad MSE")
    ax.set_ylabel("Number of samples")
    ax.set_title("1.1-Depth grad mse histogram")

    # 1.2-Depth PSNR histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(depth_psnr, bins=100)
    ax.grid()
    ax.set_xlabel("PSNR")
    ax.set_ylabel("Number of samples")
    ax.set_title("1.2-Depth PSNR histogram")

    # 1.3-Depth scatter grad mse, mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, depth_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Grad MSE")
    ax.set_ylabel("PSNR")
    ax.set_title("1.3-Depth scatter grad mse, PSNR")

    # 2-NORM EMBEDDINGS
    # 2.1-Norm cross product histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(norm_cross_prod, bins=100)
    ax.grid()
    ax.set_xlabel("Cross Product")
    ax.set_ylabel("Number of samples")
    ax.set_title("2.1-Norm cross product histogram")

    # 2.2-Depth PSNR histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(norm_psnr, bins=100)
    ax.grid()
    ax.set_xlabel("PSNR")
    ax.set_ylabel("Number of samples")
    ax.set_title("2.2-Norm PSNR histogram")

    # 2.3-Norm scatter cross product, mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(norm_cross_prod, norm_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Cross Product")
    ax.set_ylabel("PSNR")
    ax.set_title("2.3-Norm scatter cross product, PSNR")

    # 3-DEPTH-NORM COMPARISON
    # 3.1-Depth grad mse, norm mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, norm_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's grad MSE")
    ax.set_ylabel("Normal's PSNR")
    ax.set_title("3.1-Depth grad mse, norm PSNR")

    # 3.2-Depth grad mse, cross product
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, norm_cross_prod, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's grad MSE")
    ax.set_ylabel("Norm's cross product")
    ax.set_title("3.2-Depth grad mse, norm cross product")

    # 3.3-Depth PSNR, norm mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_psnr, norm_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's PSNR")
    ax.set_ylabel("Normal's PSNR")
    ax.set_title("3.3-Depth PSNR, norm PSNR")

    # 3.4-Depth PSNR, norm's cross product
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_psnr, norm_cross_prod, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's PSNR")
    ax.set_ylabel("Norm's cross product")
    ax.set_title("3.4-Depth PSNR, norm cross product")
    plt.show()


def plot_butterworth(quat, den_quat, fc_quat, period):
    # Butterworth quaternion
    wn_list = range(1, 15)
    n_list = range(1, 15)
    den_quat_butter = np.zeros([len(n_list)] + [len(wn_list)] + list(quat.shape))
    for n_ind, n in enumerate(n_list):
        for wn_ind, wn in enumerate(wn_list):
            sos = signal.butter(N=n, Wn=wn, btype='lowpass', fs=1 / period, analog=False, output='sos')
            # sos = signal.butter(N=n, Wn=wn, btype='lowpass', analog=True, output='sos')
            for i in range(4):
                den_quat_butter[n_ind, wn_ind, :, i] = signal.sosfilt(sos, quat[:, i])
    den_quat_butter = den_quat_butter / np.linalg.norm(den_quat_butter, axis=-1)[..., np.newaxis]  # Normalize

    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    plt.plot(quat[:, i], label='gt')
    plt.plot(den_quat[:, i], label=f'ideal fc_quat={fc_quat}')
    line = plt.plot(den_quat_butter[0, 0, :, i], label=f'butter')
    plt.grid()
    plt.legend()
    axcolor = 'lightgoldenrodyellow'

    # Make a vertically oriented slider to control the amplitude
    ax_n = plt.axes([0.05, 0.1, 0.0225, 0.83], facecolor=axcolor)
    n_slider = Slider(ax=ax_n,
                      label="butter order",
                      valmin=n_list[0],
                      valmax=n_list[-1],
                      valinit=n_list[0],
                      orientation="vertical")
    ax_wn = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    wn_slider = Slider(ax=ax_wn,
                       label="butter frequency",
                       valmin=wn_list[0],
                       valmax=wn_list[-1],
                       valinit=wn_list[0],
                       orientation="horizontal")

    # The function to be called anytime a slider's value changes
    def update(_):
        line[0].set_ydata(den_quat_butter[int(np.round(n_slider.val)),
                          int(np.round(wn_slider.val)), :, i])
        fig.canvas.draw_idle()

    # register the update function with each slider
    n_slider.on_changed(update)
    wn_slider.on_changed(update)
    print()


def residual_video(flight: str, double: bool, norm: bool):
    flight_path = euroc.INTERMEDIATE_DATASET_PATH + flight
    print("Processing: ", flight_path)

    # 1-LOADING
    # 1.1-Loading drone trajectory
    data = euroc.load_flight(flight)
    # data state is tf_origin_vicon0

    if double:
        # 1.2-Predict non-fa residual
        print('\nResidual model GRU CNN no embeddings')
        gru_cnn = ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING + "GRUCNNResidual_None/default/"
        gru_cnn += "20210928_090235_GRUCNNResidual_bs64_epch1000_lr1e-06_h1_p60_us50_tanh_l3_None/checkpoints/"
        gru_cnn += "epoch=999-step=584999.ckpt"

        model = DynamicalModel(param='identification', dtype=torch.float32)
        past = 60
        residual_model = GRUCNNResidual.load_from_checkpoint(gru_cnn, model=model, motor_size=6)
        residual_model.eval()

        # Presence classifier
        mlp = '/Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/PresenceClassifier_depth/default/' \
              '20211018_100951_PresenceClassifier_relu_bs256_depth_l0.0_l0.0_l3_lr1e-06_epch10000_us50_/checkpoints/' \
              'epoch=2063-step=373583.ckpt'
        from src.drone_models.residual.presence_classifier import PresenceClassifier
        classifier = PresenceClassifier.load_from_checkpoint(mlp, motor_size=6)
        classifier.eval()
        with torch.no_grad():
            label_hat = classifier(depth_emb=torch.FloatTensor(data["depth_emb"]),
                                   norm_emb=torch.FloatTensor(data["norm_emb"])).squeeze()
            label_hat = torch.round(label_hat)

        label_gt = torch.FloatTensor(data['fa_label'])
        truth_is_0 = torch.where(label_gt == 0)[0]
        truth_is_1 = torch.where(label_gt == 1)[0]
        tp_count = (label_hat[truth_is_1] == 1).sum().numpy()
        fp_count = (label_hat[truth_is_0] == 1).sum().numpy()
        tn_count = (label_hat[truth_is_0] == 0).sum().numpy()
        fn_count = (label_hat[truth_is_1] == 0).sum().numpy()
        assert tp_count + fp_count + tn_count + fn_count == len(label_gt)
        print(f'tp: {tp_count} | fp: {fp_count} | tn: {tn_count} | fn: {fn_count}')

        tp = torch.where(torch.logical_and(label_gt == 1, label_hat == 1))[0]
        fp = torch.where(torch.logical_and(label_gt == 0, label_hat == 1))[0]
        tn = torch.where(torch.logical_and(label_gt == 0, label_hat == 0))[0]
        fn = torch.where(torch.logical_and(label_gt == 1, label_hat == 0))[0]
        pcolor_classification = np.zeros_like(label_hat)
        pcolor_classification[tn] = 0
        pcolor_classification[tp] = 1
        pcolor_classification[fp] = 2
        pcolor_classification[fn] = 3

        assert (len(tp) == tp_count) and (len(fp) == fp_count) and (len(tn) == tn_count) and (len(fn) == fn_count)

        residual_hat = []
        dt = torch.FloatTensor([[1 / 200]]).repeat([past, 1])
        with torch.no_grad():
            for i in tqdm(range(len(data['motor']) - past), desc='Computing fa'):
                motor = torch.FloatTensor(data["motor"][i + 1:past + i + 1])
                state = torch.FloatTensor(data["state"][i:past + i])
                residual_hat.append(residual_model(motor=motor, state=state, dt=dt))
        residual_hat = torch.stack(residual_hat, dim=0)
        residual_gt = torch.FloatTensor(data['residual'])
        residual = (residual_gt[past:] - residual_hat).numpy()

    else:
        past = 0
        residual = data['residual']

    # 1.3-Cutting a the image frequency
    cut_img = np.unique(data["cam0"][past:])
    data_ind = np.zeros(len(cut_img), dtype=int)

    for i, img_timestamp in enumerate(cut_img):
        ind = np.abs(img_timestamp - data["timestamp"]).argmin()
        data_ind[i] = ind
    state_mean = data["state"][:, :3] + euroc.TRANS_MEAN[flight].reshape((1, 3))

    # 2-CREATING VIDEO
    # 2.1-Initialising video
    fps = 20
    camera_shape = [480, 752, 3]  # height, width, 3
    frame_shape = [1080, 1920, 3]
    dpi = 100
    saving_path = f'{ANALYSIS_DATA_VIS_EUROC_MAV}{flight}residual_video.mkv'
    mkdir_if_not_exist([f'{ANALYSIS_DATA_VIS_EUROC_MAV}{flight}'])
    codec = VideoWriter_fourcc(*'H264')
    video = VideoWriter(saving_path, codec, float(fps), (frame_shape[1], frame_shape[0]), True)  # width, height
    print(f'Video will be saved at : {saving_path}')

    for i, image_time in enumerate(tqdm(cut_img, desc="Creating a video")):
        # 2.2-Constructing image
        # 2.2.1-Camera image
        image_name = str(image_time)
        img = np.array(imageio.imread(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{image_name}.png"))
        color_frame = np.uint8(np.zeros(camera_shape))
        for j in range(3):  # color the image
            color_frame[:, :, j] = img
        img = color_frame

        # Displaying image on figure (width, height)
        fig = plt.figure(figsize=(frame_shape[1] / dpi, frame_shape[0] / dpi), dpi=dpi, facecolor='black')
        ax = fig.add_axes([0, 0.5, 1, 0.5])  # [distance to left, distance to bottom, width, height] on the figure
        ax.imshow(img)
        ax.axis('off')
        # Displaying legend: the drone position
        trans = state_mean[data_ind[i], :3]
        angles = Rotation.from_quat(data['state'][data_ind[i], 3:7]).as_euler(seq='zyx', degrees=True)[::-1]
        legend = (f'Camera position\n'
                  f'       x:{trans[0]:8.3f}m\n'
                  f'       y:{trans[1]:8.3f}m\n'
                  f'       z:{trans[2]:8.3f}m\n'
                  f'   Rot x:{angles[0]:8.3f}°\n'
                  f'   Rot y:{angles[1]:8.3f}°\n'
                  f'   Rot z:{angles[2]:8.3f}°\n'
                  f' data id:{data_ind[i]:9}\n'
                  f'frame id:{i:9}')
        ax.annotate(legend, (0.73, 0.95), xycoords='figure fraction', size=12, ha='left', va='top', family='monospace',
                    color='white')

        # 2.2.2-Plotting residual
        left, bottom = 0.03, 0.03
        ax = fig.add_axes([left, bottom, 0.99 - left, 0.49 - bottom])  # [distance to left, distance to bottom, width,
        # height] on the figure
        start, stop = data_ind[i] - 5 * 200, data_ind[i] + 5 * 200

        if norm:
            ax.plot(np.linalg.norm(residual[:, :3], axis=1), label='Forces norm')
            ax.plot(np.linalg.norm(residual[:, 3:], axis=1), label='Torques norm')
            legend = ax.legend(loc=1, bbox_to_anchor=(0.1, 1.15), frameon=False)
        else:
            ax.plot(list(range(past, len(data['t']))), residual, label=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
            legend = ax.legend(loc=1, bbox_to_anchor=(0.05, 1.35), frameon=False)
        if double:
            # Rectangle(0.2)
            # matplotlib.pat = fig.add_axes([0.2, 0.6, 1, 0.5])  # [distance to left, distance to bottom, width, height] on the figure
            color = {1: 'green', 2: 'orange', 3: 'blue'}
            for j in range(-1000, 1000):
                if data_ind[i] + j < 0 or data_ind[i] + j >= len(pcolor_classification):
                    continue
                if pcolor_classification[data_ind[i] + j] == 0:
                    continue
                y1, y2 = ax.get_ylim()
                # print(f'j: {i+j} | pcolor_classification: {pcolor_classification[i+j]} | y1: {y1} | y2: {y2}')
                c = color[pcolor_classification[data_ind[i] + j]]
                ax.fill_between(x=[data_ind[i]+j - 0.49, data_ind[i]+j + 0.49], y1=y1, y2=y2, alpha=0.1, color=[c, c])
                ax.set_ylim([y1, y2])
        ax.axvline(data_ind[i])
        [e.set_color('white') for e in legend.get_texts()]
        [e.set_fontsize(12) for e in legend.get_texts()]
        ax.grid()
        [ax.spines[e].set_color('white') for e in ['bottom', 'top', 'left', 'right']]
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('black')
        ax.set_xlim([start, stop])
        # ax.set_xlim([0, 100])

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(frame_shape)
        # plt.savefig(f'{ANALYSIS_DATA_VIS_EUROC_MAV}{flight}test.png')
        # input()
        plt.close('all')
        # frames[i] = frame
        video.write(frame[:, :, ::-1])  # Because VideoWriter.write take BGR images

    video.release()
    cv2.destroyAllWindows()


def data_cursor(flight, den_motor_dict, den_pose_dict, cam0_dict):
    cam0_sync = np.zeros(len(den_motor_dict["timestamp"]), dtype=int)  # The timestamp of the image
    for i, timestamp in enumerate(den_motor_dict["timestamp"]):
        cam0_sync[i] = cam0_dict["timestamp"][np.abs(timestamp - cam0_dict["timestamp"]).argmin()]

    fig, axs = plt.subplots(3, 1, figsize=(19.2, 10.8))
    ax = axs[0]
    ax.plot(den_pose_dict['t'] - den_motor_dict['t'][0], den_pose_dict['pose'][:, :3])
    ax.grid()
    ax = axs[1]
    ax.plot(den_motor_dict['t'] - den_motor_dict['t'][0], den_motor_dict['motor'])
    ax.grid()
    img_loaded = np.array(imageio.imread(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{cam0_sync[0]}.png"))
    axs[2].imshow(img_loaded)
    # [ax.set_xlim([140, 150]) for ax in axs]
    lines = [ax.axvline(10) for ax in axs]
    from matplotlib.widgets import Slider
    ax_n = plt.axes([0.05, 0.1, 0.0225, 0.83], facecolor='lightgoldenrodyellow')
    n_slider = Slider(ax=ax_n,
                      label="Cut val",
                      valmin=0,
                      valmax=den_motor_dict['t'][-1] - den_motor_dict['t'][0],
                      valinit=0,
                      orientation="vertical")

    def update(_):
        slider_value = n_slider.val
        lines[0].set_xdata(slider_value)
        lines[1].set_xdata(slider_value)
        fig.canvas.draw_idle()
        motor_ind = np.argmin(abs(slider_value - (den_motor_dict["t"] - den_motor_dict["t"][0])))
        img_loaded_f = np.array(imageio.imread(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{cam0_sync[motor_ind]}.png"))
        axs[2].imshow(img_loaded_f)
        # img.set_data
        print(f'Motor idx: {motor_ind}')
        print(f'cam0_dict position: {np.where(cam0_sync[motor_ind] == cam0_dict["timestamp"])[0][0]}')

    n_slider.on_changed(update)
    plt.show()
    print()


def main():
    # flight = euroc.FLIGHT_LIST[0]
    # residual_video(flight=flight, double=True, norm=False)
    # flight = euroc.FLIGHT_LIST[1]
    # residual_video(flight=flight, double=True, norm=False)
    flight = euroc.FLIGHT_LIST[2]
    residual_video(flight=flight, double=True, norm=False)


if __name__ == '__main__':
    main()
