import copy

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.misc import get_affine_transform, affine_transform


class PoseEval(object):

    def __init__(self, coco_gt):
        self.coco_gt = copy.deepcopy(coco_gt)
        cat_ids = self.coco_gt.getCatIds()
        cat = self.coco_gt.loadCats(cat_ids)[0]

        self.num_keypoints = len(cat['keypoints'])

        coco_eval = COCOeval(self.coco_gt, iouType='keypoints')
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = self.coco_gt.getImgIds()
        coco_eval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        coco_eval.params.kpt_oks_sigmas = np.ones(self.num_keypoints) * .4
        self.coco_eval = coco_eval

        self.detections = []

    def collect(self, dts, metas, adjust=.5):
        dts = dts.copy()
        ids = metas['id'].cpu().numpy()
        centers = metas['center'].cpu().numpy()
        translates = metas['translate'].cpu().numpy()
        scales = metas['scale'].cpu().numpy()

        for img_dts, img_id, center, translate, scale in zip(dts, ids, centers, translates, scales):
            trans = get_affine_transform(center, translate, scale, inv=True)
            img_dts[..., :2] = np.int64(affine_transform(img_dts[..., :2] + adjust, trans))
            mask = img_dts[..., 2] > 0
            img_dts *= mask[..., None]

            for keypoints in img_dts:
                if np.any(keypoints[..., 2]):
                    self.detections.append({
                        'image_id': int(img_id),
                        'category_id': 1,
                        'keypoints': keypoints.flatten().tolist(),
                        'score': keypoints[:, 2].mean()
                    })

    def evaluate(self):
        ap = 0.0
        if len(self.detections) > 0:
            self.coco_eval.cocoDt = COCO.loadRes(self.coco_gt, self.detections)
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
            ap = self.coco_eval.stats[0]

            self.detections = []

        return ap
