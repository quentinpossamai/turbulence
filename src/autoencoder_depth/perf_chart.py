import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

# latent_size : [Grad loss, PSNR]
autoencoder_irs_depth = {32: [2.3637, 28.98],
                         64: [3.4732, 28.53],
                         128: [5.0208, 25.95],
                         256: [1.9188, 31.38],
                         512: [0.2349, 38.14]}

if __name__ == '__main__':
    pass