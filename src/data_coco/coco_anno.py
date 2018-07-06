import os
import sys

import numpy as np

# Add COCO_PATH into PYTHON PATH
sys.path.insert(0, "../../data/cocoapi/PythonAPI")

from pycocotools.coco import COCO

from src.config.config import DataConfig


def get_kp_annotation(gtcoco, image_annos, imgpath, cfg=DataConfig):

    imginfo = gtcoco.loadImgs(image_annos[0]['image_id'])[0]

    mlist = list()
    for kp in image_annos:

        if kp['num_keypoints'] < cfg.COCO_MIN_KEYPOINT_NUM:
            continue

        bbox = np.array(kp['bbox']).astype(np.int)
        bbox_size = bbox[2]*bbox[3] #x,y,w, h
        if bbox_size < cfg.COCO_MIN_BBOX_SIZE:
            continue

        imgfilename = imginfo['file_name']
        imgwidth = imginfo['width']
        imgheight = imginfo['height']

        filename = os.path.join(imgpath, imgfilename)

        kpoints = np.array(kp['keypoints'])
        kpoints = kpoints.reshape((-1, 3))

        meta = {'filename': filename, 'width': imgwidth, 'height': imgheight,
                'keypoints': kpoints, 'bbox':bbox }
        mlist.append(meta)

    return mlist

def get_coco_kp_annotation(coco_anno_file, coco_image_path, datacfg):
    gtCoco = COCO(coco_anno_file)

    imgIds = gtCoco.getImgIds()

    xlist = []
    for imgId in imgIds:
        annIds = gtCoco.getAnnIds(imgId, iscrowd=False)
        anns = gtCoco.loadAnns(annIds)

        if len(anns) == 0:
            continue

        meta = get_kp_annotation(gtCoco, anns, coco_image_path, datacfg)
        xlist.extend(meta)

    return xlist
