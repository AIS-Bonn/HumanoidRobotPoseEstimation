import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import pycocotools
from pycocotools.coco import COCO

from utils.ops import fliplr_kpts, generate_heatmaps, generate_pafs, generate_limb_heatmaps
from utils.misc import get_affine_transform, affine_transform

cv2.setNumThreads(0)


class HRPDataset(Dataset):

    def __init__(self, cfg, root, ann_file, train, keep_ratio=False, transform=None):
        self.root = root
        self.train = train
        self.keep_ratio = keep_ratio

        self.mode = cfg.MODE
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.max_num_detections = cfg.DATASET.MAX_NUM_DETECTIONS
        self.sigma = cfg.DATASET.SIGMA
        self.num_scales = cfg.MODEL.NUM_SCALES

        self.flip = cfg.DATASET.FLIP
        self.translate = cfg.DATASET.TRANSLATE
        self.scale = cfg.DATASET.SCALE
        self.rotation = cfg.DATASET.ROTATION
        self.input_size = np.array(cfg.DATASET.INPUT_SIZE, dtype=np.uint32)
        self.output_size = np.array(cfg.DATASET.OUTPUT_SIZE, dtype=np.uint32)
        self.stride = np.int64(self.input_size / self.output_size)

        self.transform = transform

        self.coco = COCO(os.path.join(self.root, ann_file))
        cat_ids = self.coco.getCatIds()
        cat = self.coco.loadCats(cat_ids)[0]

        self.flip_order = [0, 1, 3, 2, 5, 4]
        self.limbs = np.array(cat['skeleton'], dtype=np.int64) - 1

        self.images = []
        self._load_images(cat_ids)

    def _load_images(self, cat_ids):
        for img_id in self.coco.getImgIds(catIds=cat_ids):
            image = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            mask = None
            keypoints = []
            for ann in self.coco.loadAnns(ann_ids):
                if ann['num_keypoints'] > 0:
                    keypoints.append(np.array(ann['keypoints'], dtype=np.int64).reshape(-1, 3))

                if ann['iscrowd']:
                    mask = pycocotools.mask.decode(ann['segmentation'])

            if len(keypoints) == 0:
                keypoints.append([[0, 0, 0]] * self.num_keypoints)

            filename = image['file_name']
            image_data = cv2.imread(os.path.join(self.root, 'images', filename), cv2.IMREAD_COLOR)
            if image_data is None:
                raise ValueError('Fail to read {}'.format(filename))

            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

            self.images.append({
                'id': image['id'],
                'data': image_data,
                'mask': mask,
                'keypoints': np.stack(keypoints),                    
            })
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        image_data = image['data'].copy()
        height, width = image_data.shape[:2]
        size = np.array([width, height], dtype=np.float64)
        input_size = self.input_size
        if self.keep_ratio:
            input_size = size.astype(np.uint32, copy=True)
            min_size = (self.input_size.min() + 63) // 64 * 64
            input_size[np.argsort(input_size)] = min_size, (min_size / size.min() * size.max() + 63) // 64 * 64
        center = size / 2
        in_translate = np.zeros_like(size)
        in_scale = input_size / size
        rotate = 0

        if self.train:
            flip = np.random.random()
            translate = (np.random.random(2) * 2 - 1) * self.translate
            scale = self.scale[0] + np.random.random() * (self.scale[1] - self.scale[0])
            rotate += (np.random.random() * 2 - 1) * self.rotation

            in_scale *= scale
            in_translate += translate * size * in_scale

            if flip < self.flip:
                image_data = image_data[:, ::-1]

        in_translate += size * ((input_size / size) - in_scale) / 2
        in_trans = get_affine_transform(center, in_translate, in_scale, rotate)
        input = cv2.warpAffine(image_data, in_trans, (input_size[0], input_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if self.transform is not None:
            input = self.transform(input)

        target_list, target_weight_list, keypoints_list = [], [], []
        for stride_factor in [2 ** i for i in range(self.num_scales - 1, -1, -1)]:
            mask = np.zeros(image_data.shape[:2], dtype=np.uint8) if image['mask'] is None else image['mask'].copy()
            kpts = image['keypoints'].copy().swapaxes(0, 1)
            num_robots = kpts.shape[1]
            output_size = np.uint32(input_size / (self.stride * stride_factor))
            # out_translate = np.zeros_like(size)
            # out_scale = output_size / size

            if self.train:
                # out_scale *= scale
                # out_translate += translate * size * out_scale

                if flip < self.flip:
                    mask = mask[:, ::-1]
                    kpts = fliplr_kpts(kpts, width, self.flip_order)

            # out_translate += size * ((output_size / size) - out_scale) / 2
            # out_trans = get_affine_transform(center, out_translate, out_scale, rotate)
            # out_target_weight = cv2.warpAffine(mask, out_trans, (output_size[0], output_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            in_target_weight = cv2.warpAffine(mask, in_trans, (input_size[0], input_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            out_target_weight = cv2.resize(in_target_weight, (output_size[0], output_size[1]), interpolation=cv2.INTER_LINEAR)
            target_weight = np.float32(out_target_weight == 0)  # binary
            target_weight_list.append(target_weight)

            kpts_in = kpts.copy()
            kpts_in[..., :2] = np.int64(affine_transform(kpts_in[..., :2], in_trans))
            visibility = np.all((kpts_in[..., :2] >= 0) & (kpts_in[..., :2] <= (input_size - 1)), axis=2) & (kpts_in[..., 2] > 0)
            kpts_in *= visibility[..., None]
            # kpts[..., :2] = np.int64(affine_transform(kpts[..., :2], out_trans))
            kpts[..., :2] = np.int64(affine_transform(kpts[..., :2], in_trans) / (self.stride * stride_factor))
            visibility = np.all((kpts[..., :2] >= 0) & (kpts[..., :2] <= (output_size - 1)), axis=2) & (kpts[..., 2] > 0)
            kpts *= visibility[..., None]

            # sort by scores
            if num_robots > self.max_num_detections:
                scores = np.sum(kpts[:, :, 2], axis=0)
                kpts_order = np.argsort(-scores)
                kpts = kpts[:, kpts_order]
                kpts_in = kpts_in[:, kpts_order]
                if num_robots > self.max_num_detections:
                    num_robots = self.max_num_detections

            hms = generate_heatmaps(kpts, output_size, self.sigma)
            limbs = []
            background = np.maximum(1 - np.max(hms, axis=0), 0.)
            hms = np.vstack((hms, background[None, ...]))
            if self.num_scales == 1 or self.stride[0] * stride_factor == 4:
                if self.mode == 'paf':
                    limbs = generate_pafs(kpts, self.limbs, output_size)
                elif self.mode == 'hm':
                    limbs = generate_limb_heatmaps(kpts_in, self.limbs, input_size, self.stride * stride_factor, 4 * self.sigma, thr=1.)

            target_list.append((hms, limbs))

            # padding/limiting
            keypoints = np.zeros((self.max_num_detections, self.num_keypoints, 3), dtype=np.int64)
            if num_robots > 0:
                keypoints[:num_robots] = kpts[:, :num_robots].swapaxes(0, 1)

            keypoints_list.append(keypoints)

        meta = {
            'id': image['id'],
            'keypoints': keypoints_list,
            'center': center,
            'translate': in_translate,
            'scale': in_scale
        }

        return input, target_list, target_weight_list, meta
