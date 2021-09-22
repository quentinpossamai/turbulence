import pickle
import matplotlib.pyplot as plt
import numpy as np

import src.data.euroc_mav.loading as euroc


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


def main():
    plot_y_x()


if __name__ == '__main__':
    main()
