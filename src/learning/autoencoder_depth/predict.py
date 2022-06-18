import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader

import src.utils.path as path
import src.learning.autoencoder_depth.dataloaders.irs_dataset as irs_dataset
import src.learning.autoencoder_depth.train_autoencoder_depth as auto_encoder


def get_irs_model(date: str):
    """
    Get the IRS model for the given date.
    Args:
        date: yyyymmdd_hhmmss

    Returns:
        The nn.module model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_folder = f'{path.WD_PATH}analysis/irs_encoder_decoder/EncoderDecoder/{date}/'
    print(f'Loading model from {model_folder}')
    h_prms = pickle.load(open(f'{model_folder}h_prms.pkl', 'rb'))
    model = auto_encoder.EncoderDecoder(
        activation=h_prms['activation'],
        embedding_size=h_prms['embedding_size'],
        encoding=h_prms['encoding'],
        encoder_channel=h_prms['encoder_channel'],
        encoder_layer=h_prms['encoder_layer'],
        l2=h_prms['l2'],
        lr=h_prms['lr'],
        model=h_prms['model'],
    )
    try:
        model.load_state_dict(torch.load(f'{model_folder}model.nn', map_location=device))
    except FileNotFoundError:  # New weights saving method has been implemented for recent runs.
        checkpoint = torch.load(f'{model_folder}checkpoint.pt', map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, h_prms


def plot_test_result():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = irs_dataset.IRSDataset('test', ['depth', 'normal'])
    test_dataloader = DataLoader(test_dataset, 1, False, pin_memory=True, num_workers=0)
    model_list = [
        '20220615_120420',  # 6, 152 training
        '20220613_105714',  # 6, 100
        '20220611_183523',  # 9, 32
        '20220611_183514',  # 9, 64
        '20220611_183507',  # 9, 128
        '20220611_183103',  # 6, 128
        '20220611_183013',  # 3, 128
        '20220611_182540',  # 6, 32
        '20220611_182536',  # 6, 64
        '20220611_182436',  # 3, 32
        '20220611_182353',  # 3, 64
        '20220615_180545',  # 6, 100 new
        '20220615_180535',  # 6, 152 new
        '20220615_180526',  # 6, 128 new
    ]
    model, h_prms = get_irs_model('20220615_180526')
    print(f'h_prms: {h_prms}')
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            depth_gt = batch['depth'].to(device)
            # normal = batch['normal'].to(device)
            depth_hat = model(depth_gt)
            mse = ((depth_gt - depth_hat) ** 2).mean()
            mse_grad = ((auto_encoder.compute_grad(depth_gt, device) -
                         auto_encoder.compute_grad(depth_hat, device)) ** 2).mean()
            mask = depth_gt != 0
            mse_normed = (((depth_gt - depth_hat)[mask] / depth_gt[mask]) ** 2).mean()
            depth_hat = depth_hat.squeeze().cpu().numpy()
            depth_gt = depth_gt.squeeze().cpu().numpy()

            # analysis
            fig, axs = plt.subplots(1, 2)
            axs[1].set_title(f'MSE: {mse:.4f}\n'
                             f'MSE Grad: {mse_grad:.0f}\n'
                             f'MSE Normed: {mse_normed:.4f}')
            min_d = min(depth_hat.min(), depth_gt.min())
            max_d = max(depth_hat.max(), depth_gt.max())
            axs[0].imshow(depth_gt, vmin=min_d, vmax=max_d)
            axs[1].imshow(depth_hat, vmin=min_d, vmax=max_d)
            plt.show()


def main():
    plot_test_result()


if __name__ == '__main__':
    main()
