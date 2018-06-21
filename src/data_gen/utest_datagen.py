
from mpii_datagen import MPIIDataGen
import scipy.misc
import numpy as np
import cv2
from utest_data_view import draw_joints

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
    mimage = cv2.resize(mimage, (256, 256))
    mimage += imgdata[:,:,0]

    scipy.misc.imshow(mimage)

def main_vis():
    xdata = MPIIDataGen("../../data/mpii/mpii_annotations.json",
                        "../../data/mpii/images", inres=(256, 256),
                        outres=(64, 64), is_train=True)

    count = 0
    for _img, _gthmap, _meta in xdata.generator(1, 4, sigma=1 , with_meta=True, is_shuffle=True, rot_flag=True, scale_flag=True):
        xgthmap = _gthmap[-1]
        print _img.dtype
        #scipy.misc.imshow(_img[0,:,:,:])
        debug_view_gthmap_v2(_img[0,:,:,:], xgthmap[0,:,:,:])

    print 'scan done'



if __name__ == '__main__':
    main_vis()