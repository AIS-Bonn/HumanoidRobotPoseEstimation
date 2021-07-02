"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/JointsDataset.py#L169,
https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/lib/datasets/paf.py#L18,
https://github.com/hellojialee/Improved-Body-Parts/blob/master/py_cocodata_server/py_data_heatmapper.py#L163,
https://github.com/princeton-vl/pose-ae-train/blob/master/utils/group.py#L93,
https://github.com/hellojialee/Improved-Body-Parts/blob/master/evaluate.py#L279
"""
import numpy as np
import torch
from torch.nn import functional as F


def fliplr_kpts(keypoints, width, parts_order=None):
    # horizontal flip
    keypoints[:, :, 0] = width - keypoints[:, :, 0] - 1

    # swap left & right parts
    if parts_order is not None:
        keypoints = keypoints[parts_order]

    return keypoints


def generate_heatmaps(keypoints, heatmap_size, sigma, heatmaps=None):
    num_keypoints = keypoints.shape[0]
    if heatmaps is None:
        heatmaps = np.zeros((num_keypoints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)

    mu = keypoints[..., :2]
    s = 3 * sigma + 1
    e = 3 * sigma + 2
    # check that each part of the Gaussian is in-bounds
    ul = np.int64(np.floor(mu - s))
    br = np.int64(np.ceil(mu + e))

    # generate 2D Gaussian
    x = np.arange(np.floor(-s), np.ceil(e), dtype=np.float64)
    y = x[:, np.newaxis]
    x0 = y0 = 0.0
    gaussian = 1.0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # usable Gaussian range
    g_lt = np.maximum(0, -ul)
    g_rb = np.minimum(br, heatmap_size) - ul
    # image range
    img_lt = np.maximum(0, ul)
    img_rb = np.minimum(br, heatmap_size)

    for i in range(num_keypoints):
        for j in range(keypoints.shape[1]):  # number of keypoints
            if keypoints[i, j, 2] == 0 or np.any(ul[i, j] >= heatmap_size) or np.any(br[i, j] < 0):
                continue

            heatmaps[i, img_lt[i, j, 1]:img_rb[i, j, 1], img_lt[i, j, 0]:img_rb[i, j, 0]] = np.maximum(heatmaps[i, img_lt[i, j, 1]:img_rb[i, j, 1], img_lt[i, j, 0]:img_rb[i, j, 0]], gaussian[g_lt[i, j, 1]:g_rb[i, j, 1], g_lt[i, j, 0]:g_rb[i, j, 0]])

    return heatmaps


def generate_pafs(keypoints, limbs, heatmap_size, thr=1.):
    num_limbs = limbs.shape[0]
    pafs = np.zeros((num_limbs * 2, heatmap_size[1], heatmap_size[0]), dtype=np.float32)

    for i in range(num_limbs):
        parts = keypoints[limbs[i]]
        count = np.zeros((heatmap_size[1], heatmap_size[0]), dtype=np.uint32)
        for j in range(keypoints.shape[1]):  # number of keypoints
            src, dst = parts[:, j]
            limb_vec = dst[:2] - src[:2]
            norm = np.linalg.norm(limb_vec)
            if src[2] == 0 or dst[2] == 0 or norm == 0:
                continue

            limb_vec_unit = limb_vec / norm

            min_p = np.maximum(np.round(np.minimum(src[:2], dst[:2]) - thr), 0)
            max_p = np.minimum(np.round(np.maximum(src[:2], dst[:2]) + thr), heatmap_size - 1)
            range_x = np.arange(min_p[0], max_p[0], 1, dtype=np.int64)
            range_y = np.arange(min_p[1], max_p[1], 1, dtype=np.int64)

            xx, yy = np.meshgrid(range_x, range_y)
            ba_x = xx - src[0]  # the vector from (x,y) to src
            ba_y = yy - src[1]
            limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
            mask = limb_width < thr  # mask is 2D

            tmp_vec_map = np.zeros((2, heatmap_size[1], heatmap_size[0]), dtype=np.float64)
            tmp_vec_map[:, yy, xx] = np.repeat(mask[None, ...], 2, axis=0) * limb_vec_unit[:, None, None]

            mask = np.any(np.abs(tmp_vec_map) > 0, axis=0)
            vec_map = np.multiply(pafs[2 * i:2 * (i + 1)], count[None, ...])
            vec_map += tmp_vec_map
            count[mask] += 1

            mask = (count == 0)
            count[mask] = 1

            pafs[2 * i:2 * (i + 1)] = np.divide(vec_map, count[None, ...])
            count[mask] = 0

    return pafs


def generate_limb_heatmaps(keypoints, limbs, input_size, stride, sigma, thr=1., dist_thr=.015):
    num_limbs = limbs.shape[0]
    heatmap_size = input_size // stride
    heatmaps = np.zeros((num_limbs, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    grid = np.mgrid[0:input_size[0]:stride[0], 0:input_size[1]:stride[1]] + stride[:, None, None] / 2 - 0.5

    for i in range(num_limbs):
        parts = keypoints[limbs[i]]
        count = np.zeros((heatmap_size[1], heatmap_size[0]), dtype=np.uint32)
        for j in range(keypoints.shape[1]):  # number of keypoints
            src, dst = parts[:, j]
            limb_vec = dst[:2] - src[:2]
            norm = np.linalg.norm(limb_vec)

            if src[2] == 0 or dst[2] == 0 or norm == 0:
                continue

            min_p = np.maximum(np.round((np.minimum(src[:2], dst[:2]) - (thr * stride)) / stride), 0)
            max_p = np.minimum(np.round((np.maximum(src[:2], dst[:2]) + (thr * stride)) / stride), input_size - 1)
            range_x = slice(int(min_p[0]), int(max_p[0]) + 1)
            range_y = slice(int(min_p[1]), int(max_p[1]) + 1)

            min_x = max(int(round((min(src[0], dst[0]) - (thr * stride[0])) / stride[0])), 0)
            max_x = min(int(round((max(src[0], dst[0]) + (thr * stride[0])) / stride[0])), input_size[0] - 1)
            min_y = max(int(round((min(src[1], dst[1]) - (thr * stride[1])) / stride[1])), 0)
            max_y = min(int(round((max(src[1], dst[1]) + (thr * stride[1])) / stride[1])), input_size[1] - 1)
            slice_x = slice(min_x, max_x + 1)
            slice_y = slice(min_y, max_y + 1)

            assert np.array_equal(grid[:, range_x, range_y], grid[:, slice_x, slice_y])

            deta = src[:2][:, None, None] - grid[:, range_x, range_y]
            dist = (limb_vec[0] * deta[1] - deta[0] * limb_vec[1]) / (norm + 1e-6)
            dist = np.abs(dist)
            gauss_dist = np.exp(-(dist - 0.0) ** 2 / (2 * sigma ** 2)).T
            # gauss_dist[gauss_dist <= dist_thr] = 0.01

            mask = gauss_dist > 0
            heatmaps[i, slice_y, slice_x][mask] += gauss_dist[mask]
            count[slice_y, slice_x][mask] += 1

        mask = count > 0
        heatmaps[i][mask] /= count[mask]

    return heatmaps


def get_keypoint_dets(heatmaps, nms_kernel=3, max_num_dets=10, det_thr=.1):
    # non-maximum suppression (nms)
    nms_padding = (nms_kernel - 1) // 2
    hmax = F.max_pool2d(heatmaps, nms_kernel, stride=1, padding=nms_padding)
    hmax = torch.eq(hmax, heatmaps).to(heatmaps.dtype)
    hms = heatmaps * hmax

    width = hms.size(3)
    hms = hms.view(hms.size(0), hms.size(1), -1)
    vals, inds = hms.topk(max_num_dets, dim=2, sorted=True)

    # thresholding
    inds *= torch.ge(vals, det_thr).long()

    x = inds % width
    y = (inds // width)
    dets = torch.stack((x, y), dim=3)

    return dets.cpu().numpy(), vals.cpu().numpy()


def group_dets(heatmaps, limb_heatmaps, limbs, cfg, nms_kernel=3, max_num_dets=10, det_thr=.1):
    fixed_num_midpts = cfg['NUM_MIDPOINTS']
    paf_thr = cfg['THRESHOLD']
    ignore_few_parts = cfg['IGNORE_FEW_PARTS']
    connection_ratio = cfg['CONNECTION_RATIO']
    len_rate = cfg['LENGTH_RATE']
    connection_tol = cfg['CONNECTION_TOLERANCE']
    delete_shared_parts = cfg['DELETE_SHARED_PARTS']
    min_num_connected_parts = cfg['MIN_NUM_CONNECTED_PARTS']
    min_mean_score = cfg['MIN_MEAN_SCORE']

    dets, vals = get_keypoint_dets(heatmaps, nms_kernel, max_num_dets, det_thr)
    limb_heatmaps = limb_heatmaps.cpu().numpy()
    num_images, num_keypoints = dets.shape[:2]
    image_height = limb_heatmaps.shape[2]
    outputs = np.concatenate((dets, vals[..., np.newaxis]), axis=3)
    results = np.zeros((num_images, max_num_dets, num_keypoints, 3), dtype=np.float64)

    for i in range(num_images):
        connected_limbs = []
        robot_to_part_assoc = []

        for j in range(limbs.shape[0]):
            parts_src, parts_dst = outputs[i, limbs[j]]
            part_src_type, part_dst_type = limbs[j]
            vis_parts_src = np.any(parts_src, axis=1)
            vis_parts_dst = np.any(parts_dst, axis=1)
            num_parts_src = vis_parts_src.sum()
            num_parts_dst = vis_parts_dst.sum()

            if num_parts_src > 0 and num_parts_dst > 0:
                candidates = []

                for k, src in enumerate(parts_src):
                    if not vis_parts_src[k]:
                        continue

                    for l, dst in enumerate(parts_dst):
                        if not vis_parts_dst[l]:
                            continue

                        limb_dir = dst[:2] - src[:2]
                        limb_dist = np.sqrt(np.sum(limb_dir ** 2))

                        if limb_dist == 0:
                            continue

                        # limb_dir = limb_dir / limb_dist
                        # num_midpts = fixed_num_midpts
                        num_midpts = min(int(np.round(limb_dist + 1)), fixed_num_midpts)

                        limb_midpts_coords = np.empty((2, num_midpts), dtype=np.int64)
                        limb_midpts_coords[0] = np.round(np.linspace(src[0], dst[0], num=num_midpts))
                        limb_midpts_coords[1] = np.round(np.linspace(src[1], dst[1], num=num_midpts))
                        # midpts_paf = limb_heatmaps[i, 2 * j:2 * (j + 1), limb_midpts_coords[1], limb_midpts_coords[0]]
                        # score_midpts = midpts_paf.dot(limb_dir)
                        score_midpts = limb_heatmaps[i, j, limb_midpts_coords[1], limb_midpts_coords[0]]
                        long_dist_penalty = min(0.5 * image_height / limb_dist - 1, 0)
                        connection_score = score_midpts.mean() + long_dist_penalty

                        criterion1 = np.count_nonzero(score_midpts > paf_thr) >= (connection_ratio * num_midpts)
                        criterion2 = (connection_score > 0)
                        if criterion1 and criterion2:
                            candidates.append([k, l, connection_score, limb_dist, 0.5 * connection_score + 0.25 * src[2] + 0.25 * dst[2]])

                candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
                max_connections = min(num_parts_src, num_parts_dst)
                connections = np.empty((0, 4), dtype=np.float64)

                for can in candidates:
                    if can[0] not in connections[:, 0] and can[1] not in connections[:, 1]:
                        connections = np.vstack((connections, can[:4]))
                        if len(connections) >= max_connections:
                            break

                connected_limbs.append(connections)
            else:
                connected_limbs.append([])

            for limb_info in connected_limbs[j]:
                robot_assoc_idx = []

                for robot, robot_limbs in enumerate(robot_to_part_assoc):
                    if robot_limbs[part_src_type, 0] == limb_info[0] or robot_limbs[part_dst_type, 0] == limb_info[1]:
                        robot_assoc_idx.append(robot)

                if len(robot_assoc_idx) == 1:
                    robot_limbs = robot_to_part_assoc[robot_assoc_idx[0]]
                    # if robot_limbs[part_dst_type, 0] == -1 and (robot_limbs[-1, 1] * len_rate) > limb_info[-1]:
                    if robot_limbs[part_dst_type, 0] != limb_info[1]:
                        robot_limbs[part_dst_type] = limb_info[[1, 2]]
                        robot_limbs[-2, 0] += vals[i, part_dst_type, int(limb_info[1])] + limb_info[2]
                        robot_limbs[-1, 0] += 1
                        robot_limbs[-1, 1] = max(limb_info[-1], robot_limbs[-1, 1])
                    # elif robot_limbs[part_dst_type, 0] != limb_info[1]:
                    #     if robot_limbs[part_dst_type, 1] < limb_info[2]:
                    #         if (robot_limbs[-1, 1] * len_rate) <= limb_info[-1]:
                    #             continue

                    #         robot_limbs[-2, 0] -= vals[i, int(robot_limbs[part_dst_type, 0]), int(limb_info[1])] + robot_limbs[part_dst_type, 1]
                    #         robot_limbs[part_dst_type] = limb_info[[1, 2]]
                    #         robot_limbs[-2, 0] += vals[i, part_dst_type, int(limb_info[1])] + limb_info[2]
                    #         robot_limbs[-1, 1] = max(limb_info[-1], robot_limbs[-1, 1])
                    # elif robot_limbs[part_dst_type, 0] == limb_info[1] and robot_limbs[part_dst_type, 1] <= limb_info[2]:
                    #     robot_limbs[-2, 0] -= vals[i, int(robot_limbs[part_dst_type, 0]), int(limb_info[1])] + robot_limbs[part_dst_type, 1]
                    #     robot_limbs[part_dst_type] = limb_info[[1, 2]]
                    #     robot_limbs[-2, 0] += vals[i, part_dst_type, int(limb_info[1])] + limb_info[2]
                    #     robot_limbs[-1, 1] = max(limb_info[-1], robot_limbs[-1, 1])
                elif len(robot_assoc_idx) == 2:
                    robot1_limbs = robot_to_part_assoc[robot_assoc_idx[0]]
                    robot2_limbs = robot_to_part_assoc[robot_assoc_idx[1]]
                    membership1 = (robot1_limbs[:-2, 0] >= 0)
                    membership2 = (robot2_limbs[:-2, 0] >= 0)
                    membership = (membership1 & membership2)
                    if not np.any(membership):
                        min_limb1 = np.min(robot1_limbs[:-2, 1][membership1])
                        min_limb2 = np.min(robot2_limbs[:-2, 1][membership2])
                        min_tol = min(min_limb1, min_limb2)  # min confidence
                        if limb_info[2] >= (connection_tol * min_tol) and limb_info[-1] < (robot1_limbs[-1, 1] * len_rate):
                            robot1_limbs[:-2] += robot2_limbs[:-2] + 1
                            robot1_limbs[-2, 0] += robot2_limbs[-2, 0] + limb_info[2]
                            robot1_limbs[-1, 0] += robot2_limbs[-1, 0]
                            robot1_limbs[-1, 1] = max(limb_info[-1], robot1_limbs[-1, 1])
                            robot_to_part_assoc.pop(robot_assoc_idx[1])
                    else:
                        if delete_shared_parts:
                            if limb_info[0] in robot1_limbs[:-2, 0]:
                                conn1_idx = int(np.where(robot1_limbs[:-2, 0] == limb_info[0])[0])
                                conn2_idx = int(np.where(robot2_limbs[:-2, 0] == limb_info[1])[0])
                            else:
                                conn1_idx = int(np.where(robot1_limbs[:-2, 0] == limb_info[1])[0])
                                conn2_idx = int(np.where(robot2_limbs[:-2, 0] == limb_info[0])[0])

                            assert conn1_idx != conn2_idx, "an candidate keypoint is used twice, shared by two object"

                            if limb_info[2] >= robot1_limbs[conn1_idx, 1] and limb_info[2] >= robot2_limbs[conn2_idx, 1]:
                                if robot1_limbs[conn1_idx, 1] > robot2_limbs[conn2_idx, 1]:
                                    low_conf_idx = robot_assoc_idx[1]
                                    delete_conn_idx = conn2_idx
                                else:
                                    low_conf_idx = robot_assoc_idx[0]
                                    delete_conn_idx = conn1_idx

                                robot_to_part_assoc[low_conf_idx][-2, 0] -= vals[i, int(robot_to_part_assoc[low_conf_idx][delete_conn_idx, 0]), int(limb_info[1])]
                                robot_to_part_assoc[low_conf_idx][-2, 0] -= robot_to_part_assoc[low_conf_idx][delete_conn_idx, 1]
                                robot_to_part_assoc[low_conf_idx][delete_conn_idx, 0] = -1
                                robot_to_part_assoc[low_conf_idx][delete_conn_idx, 1] = -1
                                robot_to_part_assoc[low_conf_idx][-1, 0] -= 1
                elif len(robot_assoc_idx) == 0:
                    row = np.ones((num_keypoints + 2, 2), dtype=np.float64) * -1
                    row[part_src_type] = limb_info[[0, 2]]
                    row[part_dst_type] = limb_info[[1, 2]]
                    row[-2, 0] = vals[i, part_src_type, int(limb_info[0])] + vals[i, part_dst_type, int(limb_info[1])] + limb_info[2]
                    row[-1] = [2, limb_info[-1]]
                    robot_to_part_assoc.append(row)

        if ignore_few_parts:
            robots_to_delete = []
            for robot_id, robot_info in enumerate(robot_to_part_assoc):
                if robot_info[-1, 0] < min_num_connected_parts or robot_info[-2, 0] / robot_info[-1, 0] < min_mean_score:
                    robots_to_delete.append(robot_id)

            for index in robots_to_delete[::-1]:
                robot_to_part_assoc.pop(index)

        det_idx = 0
        for robot in sorted(robot_to_part_assoc, key=lambda x: x[-1, 0], reverse=True):
            for k in range(num_keypoints):
                idx = robot[k, 0]
                if idx > -1:
                    results[i, det_idx, k] = np.append(dets[i, k, int(idx)], robot[k, 1])

            det_idx += 1
            if det_idx >= max_num_dets:
                break

    return results, dets


def aggregate_multi_scale(outputs, num_scales, num_channels=None, output_size=None):
    count = 0
    scales = []
    for i in range(len(outputs) -1, -1, -1):
        if count < num_scales and outputs[i] is not None:
            outputs_scale = outputs[i][:, -1] if num_channels is None else outputs[i][:, -1, :num_channels]
            if count == 0 and output_size is None:
                output_size = (outputs_scale.size(-2), outputs_scale.size(-1))
                scales.append(outputs_scale)
            else:
                scales.append(F.interpolate(outputs_scale, size=output_size, mode='bilinear', align_corners=False))
            count += 1

    return torch.stack(scales).mean(0)
