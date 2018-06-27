import os
import sys
import random
import json
import scipy
import scipy.misc
import data_process
import numpy as np

# Most of functions in this file are adpoted from https://github.com/bearpaw/pytorch-pose
# with minor changes to fit Keras

def load_sample_ids(jsonfile, is_train):
    # create train/val split
    with open(jsonfile) as anno_file:
        anno = json.load(anno_file)

    val_anno, train_anno = [], []
    for idx, val in enumerate(anno):
        if val['isValidation'] == True:
            val_anno.append(anno[idx])
        else:
            train_anno.append(anno[idx])

    if is_train:
        return train_anno
    else:
        return val_anno

def draw_joints(cvmat, joints):
    # fixme: image load by scipy is RGB, not CV2's channel BGR
    import cv2
    for _joint in joints:
        _x, _y, _visibility = _joint
        if _visibility == 1.0:
           cv2.circle(cvmat, center=(int(_x), int(_y)), color=(1.0, 1.0, 0), radius=7, thickness=2)


def generate_gt_map(joints, sigma, outres):
    npart = joints.shape[0]
    gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
    for i in range(npart):
        gtmap[:, :, i] = data_process.draw_labelmap(gtmap[:,:,i], joints[i,:], sigma)
    return gtmap

def view_crop_image(anno):

    print anno.keys()

    img_paths = anno['img_paths']
    img_width = anno['img_width']
    img_height = anno['img_height']

    #print anno.keys()

    imgdata = scipy.misc.imread(os.path.join("../../data/mpii/images", img_paths))
    draw_joints(imgdata, anno['joint_self'])
    #scipy.misc.imshow(imgdata)

    center = np.array(anno['objpos'])
    outimg = data_process.crop(imgdata, center= center,  scale=anno['scale_provided'], res=(256, 256), rot=0)
    outimg_normalized = data_process.normalize(outimg)
 
    print outimg.shape

    newjoints = data_process.transform_kp(np.array(anno['joint_self']), center, anno['scale_provided'], (64, 64), rot=0)
    #draw_joints(outimg_normalized, newjoints.tolist())
    #scipy.misc.imshow(outimg_normalized)

    '''
    mimage = np.zeros(shape=(64, 64), dtype=np.float)
    gtmap = generate_gt_map(newjoints, sigma=1, outres=(64, 64))
    for i in range(16):
        mimage += gtmap[:, :, i]
    scipy.misc.imshow(mimage)
    '''

    # meta info
    metainfo = []
    orgjoints = np.array(anno['joint_self'])
    for i in range(newjoints.shape[0]):
        meta = {'center': center, 'scale': anno['scale_provided'], 'pts': orgjoints[i], 'tpts': newjoints[i]}
        metainfo.append(meta)

    # transform back
    tpbpts = list()
    for i in range(newjoints.shape[0]):
        tpts = newjoints[i]
        meta = metainfo[i]
        orgpts = tpts
        orgpts[0:2] = data_process.transform(tpts, meta['center'], meta['scale'], res=[64, 64], invert=1, rot=0)
        tpbpts.append(orgpts)

    print tpbpts

    draw_joints(imgdata, np.array(tpbpts))
    scipy.misc.imshow(imgdata)


def main():
    annolst = load_sample_ids("../../data/mpii/mpii_annotations.json", is_train=False)
    for _anno in annolst:
        view_crop_image(_anno)


def get_kp_count():
    matchedParts = { \
        'ankle': [0, 5],  # ankle
        'knee': [1, 4],  # knee
        'hip': [2, 3],  # hip
        'wrist' : [10, 15],  # wrist
        'elbow' : [11, 14],  # elbow
        'shoulder' : [12, 13],  # shoulder
        'plevis': [6],
        'thorax': [7],
        'upper_neck' : [8],
        'head_top' : [9]
    }

    annolst = load_sample_ids("../../data/mpii/mpii_annotations.json", is_train=False)

    sum = np.zeros((16), dtype=np.float)
    for _anno in annolst:
        joints = np.array(_anno['joint_self'])
        visiblekp = joints[:,2]
        sum += visiblekp

    print sum

    x = {key:0 for key in matchedParts.keys()}

    for key in matchedParts.keys():
        for i in matchedParts[key]:
            x[key] += sum[i]
        x[key] = x[key] / len(matchedParts[key])
    print x

if __name__ == '__main__':
    get_kp_count()