
from mpii_datagen import MPIIDataGen
import scipy.misc
import numpy as np
import cv2

def debug_view_gthmap(gthmap):
    mimage = np.zeros(shape=(gthmap.shape[0], gthmap.shape[1]), dtype=np.float)
    print mimage.shape, gthmap.shape
    for i in range(gthmap.shape[-1]):
        mimage += gthmap[:, :, i]
    scipy.misc.imshow(mimage)


def main_vis():
    xdata = MPIIDataGen("../../data/mpii/mpii_annotations.json",
                        "../../data/mpii/images", inres=(256, 256),
                        outres=(64, 64), is_train=True)

    count = 0
    for _img, _gthmap in xdata.generator(1, sigma=1 , is_shuffle=False):
        #scipy.misc.imshow(_img[0,:,:,:])
        debug_view_gthmap(_gthmap[0,:,:,:])

    print 'scan done'

if __name__ == '__main__':
    main_vis()