
import os
import sys
import json
import copy
import numpy as np
import cv2

# Add COCO_PATH into PYTHON PATH
sys.path.insert(0, "../data_coco")
sys.path.insert(0, "../../cocoapi/PythonAPI")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data_coco.coco_anno import get_coco_kp_annotation

from src.data_coco.preprocess import crop_image, rotate_image



def get_crop_area(bbox, imgwidth, imgheight):

    scale = 1.1
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_center_x, bbox_center_y = bbox[0] + bbox[2]//2 , bbox[1] + bbox[3]//2

    ratio = 256.0/192
    if h/w > ratio:
        w = int(h/ratio)
    else:
        h = int(w*ratio)

    top_x = max(0, bbox_center_x - w * scale//2 )
    top_y = max(0, bbox_center_y - h * scale//2 )
    bottom_x = min(imgwidth, bbox_center_x + w * scale//2)
    bottom_y = min(imgheight, bbox_center_y + h * scale//2)

    crop_box = [top_x, top_y, bottom_x-top_x, bottom_y - top_y]
    return np.array(crop_box).astype(np.int)


def process_image(kp_anno):

    cvmat, mkpoints = crop_image(kp_anno)

    cvmat, mkpoints = rotate_image(cvmat, mkpoints, 30)

    for i in range(mkpoints.shape[0]):
        v = mkpoints[i,2]
        if v > 0:
            cv2.circle(cvmat, (mkpoints[i,0], mkpoints[i,1]), radius=7, color=(0, 0, 255))

    cv2.imshow('img', cvmat)
    cv2.waitKey()

def visual_kp_annotation(kp_anno):

    filename = kp_anno['filename']
    ximg = cv2.imread(filename)

    kp = kp_anno['keypoints']

    bbox = kp_anno['bbox'].astype(np.int)

    cv2.rectangle(ximg, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=(255, 0, 0))
    for i in range(kp.shape[0]):
        v = kp[i,2]
        if v > 0:
            cv2.circle(ximg, (kp[i,0], kp[i,1]), radius=7, color=(0, 0, 255))

    cv2.imshow('img', ximg)
    cv2.waitKey()


def main():
    train_anno = '/home/yli150/coco_data/data/coco_2017/annotations/person_keypoints_train2017.json'
    annlist = get_coco_kp_annotation(train_anno, '../../data/coco_2017/train2017')
    print len(annlist)
    for kpann in annlist:
        process_image(kpann)


if __name__ == "__main__":
    main()