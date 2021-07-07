"""
Implementation of calculus with quaternions with pytorch.
Source: Graf, Basile. ‘Quaternions and Dynamics’. ArXiv:0811.2889 [Math-Ph],
        18 November 2008. https://arxiv.org/abs/0811.2889.
Quaternion is in x, y, z, w shape.
"""
import numpy as np
import random
from scipy.spatial.transform import Rotation
import torch


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    Quaternion is in x, y, z, w convention.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(q.view(-1, 4, 1), r.view(-1, 1, 4))

    w = terms[:, 3, 3] - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2]
    x = terms[:, 0, 3] + terms[:, 3, 0] + terms[:, 1, 2] - terms[:, 2, 1]
    y = terms[:, 1, 3] + terms[:, 3, 1] + terms[:, 2, 0] - terms[:, 0, 2]
    z = terms[:, 2, 3] + terms[:, 3, 2] + terms[:, 0, 1] - terms[:, 1, 0]
    return torch.stack((x, y, z, w), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions. q in x, y, z, w convention.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, :-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, -1:] * uv + uuv)).view(original_shape)


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, 4), where L is the sequence length.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 2
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=-1)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


# Numpy-backed implementations
def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()


# def qmul_old_convention(q, r):
#     """
#     Multiply quaternion(s) q with quaternion(s) r.
#     Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
#     Returns q*r as a tensor of shape (*, 4).
#     """
#     assert q.shape[-1] == 4
#     assert r.shape[-1] == 4
#
#     original_shape = q.shape
#
#     # Compute outer product
#     terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
#
#     w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
#     x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
#     y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
#     z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
#     return torch.stack((w, x, y, z), dim=1).view(original_shape)
#
#
# def qrot_origin(q, v):
#     """
#     Rotate vector(s) v about the rotation described by quaternion(s) q.
#     Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
#     where * denotes any number of dimensions.
#     Returns a tensor of shape (*, 3).
#     """
#     assert q.shape[-1] == 4
#     assert v.shape[-1] == 3
#     assert q.shape[:-1] == v.shape[:-1]
#
#     original_shape = list(v.shape)
#     q = q.view(-1, 4)
#     v = v.view(-1, 3)
#
#     qvec = q[:, 1:]
#     uv = torch.cross(qvec, v, dim=1)
#     uuv = torch.cross(qvec, uv, dim=1)
#     return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


# def qeuler(q, order, epsilon=0):
#     """
#     Convert quaternion(s) q to Euler angles.
#     Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
#     Returns a tensor of shape (*, 3).
#     """
#     assert q.shape[-1] == 4
#
#     original_shape = list(q.shape)
#     original_shape[-1] = 3
#     q = q.view(-1, 4)
#
#     q0 = q[:, 0]
#     q1 = q[:, 1]
#     q2 = q[:, 2]
#     q3 = q[:, 3]
#
#     if order == 'xyz':
#         x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
#         y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
#         z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
#     elif order == 'yzx':
#         x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
#         y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
#         z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
#     elif order == 'zxy':
#         x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
#         y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
#         z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
#     elif order == 'xzy':
#         x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
#         y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
#         z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
#     elif order == 'yxz':
#         x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
#         y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
#         z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
#     elif order == 'zyx':
#         x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
#         y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
#         z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
#     else:
#         raise
#
#     return torch.stack((x, y, z), dim=1).view(original_shape)


# def qeuler_np(q, order, epsilon=0, use_gpu=False):
#     if use_gpu:
#         q = torch.from_numpy(q).cuda()
#         return qeuler(q, order, epsilon).cpu().numpy()
#     else:
#         q = torch.from_numpy(q).contiguous()
#         return qeuler(q, order, epsilon).numpy()


# def expmap_to_quaternion(e):
#     """
#     Convert axis-angle rotations (aka exponential maps) to quaternions.
#     Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
#     Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
#     Returns a tensor of shape (*, 4).
#     """
#     assert e.shape[-1] == 3
#
#     original_shape = list(e.shape)
#     original_shape[-1] = 4
#     e = e.reshape(-1, 3)
#
#     theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
#     w = np.cos(0.5 * theta).reshape(-1, 1)
#     xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
#     return np.concatenate((w, xyz), axis=1).reshape(original_shape)


# def euler_to_quaternion(e, order):
#     """
#     Convert Euler angles to quaternions.
#     """
#     assert e.shape[-1] == 3
#
#     original_shape = list(e.shape)
#     original_shape[-1] = 4
#
#     e = e.reshape(-1, 3)
#
#     x = e[:, 0]
#     y = e[:, 1]
#     z = e[:, 2]
#
#     rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
#     ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
#     rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)
#
#     result = None
#     for coord in order:
#         if coord == 'x':
#             r = rx
#         elif coord == 'y':
#             r = ry
#         elif coord == 'z':
#             r = rz
#         else:
#             raise
#         if result is None:
#             result = r
#         else:
#             result = qmul_np(result, r)
#
#     # Reverse antipodal representation to have a non-negative "w"
#     if order in ['xyz', 'yzx', 'zxy']:
#         result *= -1
#
#     return result.reshape(original_shape)

# def _test():
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#
#     n = 1000
#     tmp = torch.randn(n, 4, dtype=float)
#     quat_a = (tmp / torch.linalg.norm(tmp, axis=1).unsqueeze(1)).clone().detach()
#     tmp = torch.randn(n, 4, dtype=float)
#     quat_b = (tmp / torch.linalg.norm(tmp, axis=1).unsqueeze(1)).clone().detach()
#
#     vect = torch.randn(n, 3, dtype=float)
#
#     quat_origin_a = torch.clone(quat_a[:, [3, 0, 1, 2]])
#     quat_origin_b = torch.clone(quat_b[:, [3, 0, 1, 2]])
#
#     assert (quat_a == quat_origin_a[:, [1, 2, 3, 0]]).all()
#
#     res = qmul(quat_a, quat_b)
#     res_origin = qmul_old_convention(quat_origin_a, quat_origin_b)
#     assert torch.isclose(res, res_origin[:, [1, 2, 3, 0]]).all()
#
#     res1 = qrot(quat_a, vect)
#     res1_origin = qrot_origin(quat_origin_a, vect)
#     assert torch.isclose(res1, res1_origin).all()
#
#     rot_a = torch.from_numpy(Rotation.from_quat(quat_a).as_matrix())
#     roted_vect = (rot_a @ vect.unsqueeze(-1)).squeeze()
#     assert torch.isclose(res1, roted_vect).all()


if __name__ == '__main__':
    pass
    # _test()
