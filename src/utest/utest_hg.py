import os
from src.net.hg_blocks import create_hourglass_network, bottleneck_block, bottleneck_mobile
from src.net.hourglass import HourglassNet
import numpy as np
import scipy.misc
from src.data_coco.coco_datagen import CocoDataGen
from src.config.config import AllConfig
from src.data_coco.preprocess import draw_labelmap
from src.eval.heatmap_process import post_process_heatmap


def view_predict_hmap(predout, show_raw=False):
    gtmap = predout[-1]
    gtmap = gtmap[0,:,:,:]

    if show_raw:
        mimage = gtmap
    else:
        mimage = np.zeros(shape=gtmap.shape, dtype=np.float)
        kplst = post_process_heatmap(gtmap)
        for i, kpoint in enumerate(kplst):
            mimage[:,:,i] = draw_labelmap(mimage[:,:,i], kpoint, sigma=2)

    sumimage = np.zeros(shape=(64, 48))
    for i in range(gtmap.shape[-1]):
        sumimage += mimage[:,:,i]
    scipy.misc.imshow(sumimage)


def main_test():
    defaultcfg =  AllConfig()
    xnet = HourglassNet(cfg=defaultcfg)

    xnet.load_model("../../trained_models/hg_s2_b1/net_arch.json",
                    "../../trained_models/hg_s2_b1/weights_epoch29.h5")

    valdata = CocoDataGen(defaultcfg.datacfg, train=False)

    for _img, _gthmap in valdata.generator(1, 8, sigma=1, is_shuffle=False):
        out = xnet.model.predict(_img)

        # bgr -> rgb
        showimg = _img[0,:,:,:]
        showimg = showimg[..., ::-1]
        scipy.misc.imshow(showimg)
        #view_predict_hmap(_gthmap)
        view_predict_hmap(out, show_raw=True)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main_test()