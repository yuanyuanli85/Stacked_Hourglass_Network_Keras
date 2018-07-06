
import numpy as np


class NetConfig(object):
    STACK_NUM   = 2
    MOBILE      = False
    CLASS_NUM   = 17

class TrainConfig(object):
    LEARNING_RATE = 5e-4
    BATCH_SIZE    = 8

class DataConfig(object):

    COCO_TRAIN_ANNO_JSON  = '../../data/coco_2017/annotations/person_keypoints_train2017.json'
    COCO_VAL_ANNO_JSON    = '../../data/coco_2017/person_keypoints_val2017.json'
    COCO_TRAIN_IMAGE_PATH = "../../data/coco_2017/train2017"
    COCO_VAL_IMAGE_PATH = "../../data/coco_2017/val2017"

    COCO_MIN_KEYPOINT_NUM = 6
    COCO_MIN_BBOX_SIZE = 2000

    COCO_CHANNEL_MEAN = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR

    IMAGE_HEIGHT = 256
    IMAGE_WIDTH  = 192

    NETWORK_OUT_HEIGHT = 64
    NETWORK_OUT_WIDTH  = 48

    PART_NUM = 17

    CROP_SCALE = 1.1
    ASPECT_RATIO = IMAGE_HEIGHT*1.0/IMAGE_WIDTH

    COCO_KP_ANNO_KEYS=[u'nose', u'left_eye', u'right_eye', u'left_ear', u'right_ear', \
                       u'left_shoulder', u'right_shoulder', u'left_elbow', u'right_elbow', \
                       u'left_wrist', u'right_wrist', u'left_hip', u'right_hip', \
                       u'left_knee', u'right_knee', u'left_ankle', u'right_ankle']

                           #eyes,  ear ,   shoulder  elbow   wrist    hip       knee     ankle
    COCO_SYMMETRY_PARIS = [(1, 2), (3, 4), (5, 6) , (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

class AllConfig(object):
    datacfg = DataConfig()
    netcfg  = NetConfig()
    traincfg = TrainConfig()