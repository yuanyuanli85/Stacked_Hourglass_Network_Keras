
import sys
sys.path.insert(0, "../data_coco")

from coco_datagen import CocoDataGen
from config import DataConfig
import scipy.misc
import numpy as np
import cv2

def debug_view_gthmap(gthmap):
    mimage = np.zeros(shape=(gthmap.shape[0], gthmap.shape[1]), dtype=np.float)
    print mimage.shape, gthmap.shape
    for i in range(gthmap.shape[-1]):
        mimage += gthmap[:, :, i]
    scipy.misc.imshow(mimage)

def debug_view_gthmap_v2(imgdata, gthmap):
    mimage = np.zeros(shape=(gthmap.shape[0], gthmap.shape[1]), dtype=np.float)
    print mimage.shape, gthmap.shape
    for i in range(gthmap.shape[-1]):
        mimage += gthmap[:, :, i]
    mimage = cv2.resize(mimage, (192, 256))
    mimage += imgdata[:,:,0]
    scipy.misc.imshow(mimage)


def main_vis():
    xdata = CocoDataGen(cfg=DataConfig, train=True )

    count = 0
    for _img, _gthmap, _meta in xdata.generator(1, 4, sigma=1 , with_meta=True, is_shuffle=True, rot_flag=True, scale_flag=True):
        xgthmap = _gthmap[-1]
        ximg = _img[0,:,:,:]
        #scipy.misc.imshow(_img[0,:,:,:])
        debug_view_gthmap_v2(_img[0,:,:,:], xgthmap[0,:,:,:])
        #debug_view_gthmap(xgthmap[0,:,:,:])

    print 'scan done'



if __name__ == '__main__':
    main_vis()