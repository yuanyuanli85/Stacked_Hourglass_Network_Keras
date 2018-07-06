
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


class AllConfig(object):
    datacfg = DataConfig()
    netcfg  = NetConfig()
    traincfg = TrainConfig()