import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import torch

from src.data.blackbird.blackbird_loading import BlackbirdDataset


plt.style.use('seaborn')

DT = (1 / 360)
M = 0.913
IX = 4.9e-2
IY = 4.9e-2
IZ = 6.9e-2
L = 0.13

if __name__ == '__main__':
    data = BlackbirdDataset("train").data
    time = data[:, 0]
    state = data[:, 1:13]
    motor = data[:, 13:17]
    imu = data[:, 17:23]
    next_time = data[:, 23:24]
    dstate = data[:, 24:36]
    next_state = data[:, 36:]

    p, q, v, r, m = [], [], [], [], []
    future = []
    datadir = "Dataset/clean_data"
    if not os.path.isdir(datadir):
        datadir = "/home/sjanny/Drone_dataset/clean_data/"
    for data_name in tqdm(os.listdir(datadir)):
        state_gt = np.load(datadir + data_name)[6000:-6000]
        # state_gt = state_gt[np.arange(0, state_gt.shape[0], DIV), :]
        trajectory = torch.FloatTensor(state_gt[:, :13])
        rpm = torch.FloatTensor(state_gt[:, 13:])
        p.append(trajectory[:-1, :3])
        q.append(trajectory[:-1, 3:7])
        v.append(trajectory[:-1, 7:10])
        r.append(trajectory[:-1, 10:13])
        m.append(rpm[:-1])
        future.append(trajectory[1:, :13])

    position = torch.cat(p, dim=0)
    quaternion = torch.cat(q, dim=0)
    velocity = torch.cat(v, dim=0)
    rpy_rate = torch.cat(r, dim=0)
    rpm = torch.cat(m, dim=0)
    future = torch.cat(future, dim=0)

    Pt1 = torch.FloatTensor([[-1, +1, +1, -1],
                             [-1, -1, +1, +1],
                             [0, 0, 0, 0]]).view(1, 3, 4)

    Pt2 = torch.FloatTensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, -1, 1, -1]]).view(1, 3, 4)
    Pf = Pt2 ** 2
    memory = 100
    for p in tqdm(list(itertools.permutations([0, 1, 2, 3]))):
        Omega = rpm.unsqueeze(-1) ** 2
        Omega = Omega[:, p]
        B = Omega.shape[0]
        gravity = torch.FloatTensor([0, 0, -9.81]).view(-1, 3)
        ## LINEAR PFD
        R = torch.FloatTensor(Rotation.from_quat(quaternion).as_matrix())
        A_position = torch.cat([DT ** 2 / M * R @ Pf @ Omega, torch.zeros(B, 3, 1)], dim=-1)
        A_speed = torch.cat([DT / M * R @ Pf @ Omega, torch.zeros(B, 3, 1)], dim=-1)

        b_position = position + DT / M * velocity + DT ** 2 * gravity
        b_speed = velocity + DT * gravity

        ## ANGULAR PFD
        I = torch.FloatTensor([IX, IY, IZ]).view(1, 3, 1)

        qx, qy, qz, qw = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        Q_1 = torch.stack([qw, -qz, qy], dim=-1)
        Q_2 = torch.stack([qz, qw, -qx], dim=-1)
        Q_3 = torch.stack([-qy, qx, qw], dim=-1)
        Q_4 = torch.stack([-qx, -qy, qz], dim=-1)
        Q = torch.stack([Q_1, Q_2, Q_3, Q_4], dim=1)

        A_rpy = torch.cat([DT * (1 / I) * L / np.sqrt(2) * Pt1 @ Omega, DT * (1 / I) * Pt2 @ Omega], dim=-1)
        A_quat = 1 / 2 * DT * Q @ A_rpy

        b_rpy = rpy_rate - DT * (1 / I).squeeze(-1) * torch.cross(rpy_rate, rpy_rate * I.squeeze(-1))
        b_quat = quaternion + DT / 2 * (Q @ b_rpy.unsqueeze(-1)).squeeze(-1)

        A = torch.cat([A_position, A_quat, A_speed, A_rpy], dim=1)
        b = torch.cat([b_position, b_quat, b_speed, b_rpy], dim=-1)
        y = future

        b = b.view(-1)
        A = A.view(-1, 2)
        y = y.reshape(-1)
        resultat = torch.pinverse(A) @ (y - b)

        error = ((A @ resultat - (y - b)) ** 2).mean()
        if error < memory and resultat[0] > 0 and resultat[1] > 0:
            memory = error