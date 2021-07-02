import numpy as np


def get_affine_transform(center, translate, scale, rotate=0, inv=False):
    rotate_rad = np.pi * rotate / 180
    cs, sn = np.cos(rotate_rad), np.sin(rotate_rad)

    # M = T * C * RS * C^-1
    transform = np.zeros((3, 3), dtype=np.float32)
    transform[0, 0] = cs
    transform[1, 1] = cs
    transform[0, 1] = -sn
    transform[1, 0] = sn
    transform[:2, :2] *= scale
    if rotate != 0:
        transform[0, 2] = np.sum(transform[0, :2] * -center)
        transform[1, 2] = np.sum(transform[1, :2] * -center)
        transform[:2, 2] += center * scale
    transform[:2, 2] += translate
    transform[2, 2] = 1

    if inv:
        transform = np.linalg.pinv(np.vstack([transform[:2], [0, 0, 1]]))

    return transform[:2]


def affine_transform(src_pts, trans):
    src_pts_reshaped = src_pts.reshape(-1, 2)
    dst_pts = np.dot(np.concatenate((src_pts_reshaped, np.ones((src_pts_reshaped.shape[0], 1))), axis=1), trans.T)

    return dst_pts.reshape(src_pts.shape)
