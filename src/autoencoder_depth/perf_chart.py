import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

# latent_size : [Grad loss, PSNR]

depth_irs_irs = {32: [3.4532, 27.34],
                 64: [4.0321, 27.64],
                 128: [5.0623, 25.76],
                 256: [2.7115, 30.11],
                 512: [1.4701, 33.30]}

depth_irs_eurocmav = {32: [0.0382, 27.02],
                      64: [0.0260, 28.44],
                      128: [0.0492, 25.62],
                      256: [0.0240, 32.05],
                      512: [0.0307, 32.52], }

if __name__ == '__main__':

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].set_title("Depth2Depth IRS->IRS")
    ax[1].set_title("Depth2Depth IRS->EurocMav")
    for k in depth_irs_irs.keys():
        x_irs, y_irs = depth_irs_irs[k]
        x_euroc, y_euroc = depth_irs_eurocmav[k]
        ax[0].scatter(x_irs, y_irs, c="black")
        ax[1].scatter(x_euroc, y_euroc, c="black")

        ax[0].text(x_irs * 1.005, y_irs * 1.005, str(k))
        ax[1].text(x_euroc * 1.005, y_euroc * 1.005, str(k))

        ax[0].set_ylim(25, 35)
        ax[0].set_xlim(0, 6)
        ax[1].set_ylim(25, 35)
        ax[1].set_xlim(0, 0.06)

    ax[0].set_xlabel("MSE on Gradient")
    ax[0].set_ylabel("PSNR")
    ax[1].set_xlabel("MSE on Gradient")
    ax[1].set_ylabel("PSNR")
    fig.tight_layout()
    plt.show()
