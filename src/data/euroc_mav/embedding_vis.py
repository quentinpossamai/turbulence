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
    norm_mse = np.array([e["MSE"] for e in norm_emb])

    # 1-DEPTH EMBEDDINGS
    # 1.1-Depth grad mse histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(depth_grad_mse, bins=100)
    ax.grid()
    ax.set_xlabel("Grad MSE")
    ax.set_ylabel("Number of samples")
    ax.set_title("1.1-Depth grad mse histogram")

    # 1.2-Depth mse histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(depth_mse, bins=100)
    ax.grid()
    ax.set_xlabel("MSE")
    ax.set_ylabel("Number of samples")
    ax.set_title("1.2-Depth mse histogram")

    # 1.3-Depth scatter grad mse, mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, depth_mse, s=0.5)
    ax.grid()
    ax.set_xlabel("Grad MSE")
    ax.set_ylabel("MSE")
    ax.set_title("1.3-Depth scatter grad mse, mse")

    # 2-NORM EMBEDDINGS
    # 2.1-Norm cross product histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(norm_cross_prod, bins=100)
    ax.grid()
    ax.set_xlabel("Cross Product")
    ax.set_ylabel("Number of samples")
    ax.set_title("2.1-Norm cross product histogram")

    # 2.2-Depth mse histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(norm_mse, bins=100)
    ax.grid()
    ax.set_xlabel("MSE")
    ax.set_ylabel("Number of samples")
    ax.set_title("2.2-Depth mse histogram")

    # 2.3-Norm scatter cross product, mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(norm_cross_prod, norm_mse, s=0.5)
    ax.grid()
    ax.set_xlabel("Cross Product")
    ax.set_ylabel("MSE")
    ax.set_title("2.3-Norm scatter cross product, mse")

    # 3-DEPTH-NORM COMPARISON
    # 3.1-Depth grad mse, norm mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, norm_mse, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's grad MSE")
    ax.set_ylabel("Normal's MSE")
    ax.set_title("3.1-Depth grad mse, norm mse")

    # 3.2-Depth grad mse, cross product
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, norm_cross_prod, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's grad MSE")
    ax.set_ylabel("Norm's cross product")
    ax.set_title("3.2-Depth grad mse, cross product")

    # 3.3-Depth mse, norm mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_mse, norm_mse, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's MSE")
    ax.set_ylabel("Normal's MSE")
    ax.set_title("3.3-Depth mse, norm mse")

    # 3.4-Depth mse, cross product
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_mse, norm_cross_prod, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's MSE")
    ax.set_ylabel("Norm's cross product")
    ax.set_title("3.4-Depth mse, cross product")
    plt.show()


def main():
    plot_y_x()


if __name__ == '__main__':
    main()
